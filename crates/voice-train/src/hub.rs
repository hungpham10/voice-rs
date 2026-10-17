//! Đẩy kết quả huấn luyện lên HuggingFace Hub — **mặc định tắt** (§13.6.4a).
//!
//! Người dùng bật bằng `--push-to-hub` + `--hub-repo <user/name>`; token lấy từ
//! `--hf-token` hoặc biến môi trường `HF_TOKEN`.
//!
//! Cài đặt dùng `curl` (HTTP multipart) thay vì kéo thêm crate mạng: đây là tiện
//! ích của công cụ huấn luyện, không thuộc runtime suy luận. Repo đích **phải đã
//! tồn tại** trên Hub; ta chỉ tải file lên nhánh `main`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Cấu hình Hub đã kiểm hợp lệ.
#[derive(Debug, Clone)]
pub struct HubConfig {
    pub repo: String,
    pub token: String,
    /// Tiền tố đường dẫn trong repo (rỗng = để ở gốc).
    pub prefix: String,
}

impl HubConfig {
    /// Trả `Ok(None)` khi không bật push; khi bật thì **bắt buộc** có repo + token.
    pub fn resolve(
        push: bool,
        repo: Option<String>,
        token: Option<String>,
        prefix: String,
    ) -> Result<Option<Self>, String> {
        if !push {
            return Ok(None);
        }
        let repo = repo.ok_or("--push-to-hub cần --hub-repo <user/name>")?;
        let token = token
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .filter(|t| !t.is_empty())
            .ok_or("--push-to-hub cần token: đặt env HF_TOKEN hoặc --hf-token")?;
        Ok(Some(Self {
            repo,
            token,
            prefix,
        }))
    }

    fn repo_path(&self, file: &Path) -> Result<String, String> {
        let name = file
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("tên file không hợp lệ: {}", file.display()))?;
        Ok(if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}/{}", self.prefix.trim_end_matches('/'), name)
        })
    }

    /// Tải từng file (bỏ qua file không tồn tại) lên nhánh `main`.
    pub fn push(&self, files: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
        for f in files {
            if !f.exists() {
                continue;
            }
            let repo_path = self.repo_path(f)?;
            self.upload_one(f, &repo_path)?;
            println!("  ↺ đã đẩy {} → {}/{}", f.display(), self.repo, repo_path);
        }
        Ok(())
    }

    fn upload_one(&self, local: &Path, repo_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!(
            "https://huggingface.co/api/models/{}/upload/main/{}",
            self.repo, repo_path
        );
        let auth = format!("Authorization: Bearer {}", self.token);
        let form = format!("file=@{}", local.display());

        let out = Command::new("curl")
            .args([
                "-sS",
                "--fail-with-body",
                "-X",
                "POST",
                "-H",
                &auth,
                "-F",
                &form,
                &url,
            ])
            .output()
            .map_err(|e| format!("không chạy được `curl` (cần cài curl): {e}"))?;

        if !out.status.success() {
            let body = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "upload {} lên {} lỗi ({}): {}",
                local.display(),
                self.repo,
                out.status,
                body.trim()
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_yields_none() {
        assert!(HubConfig::resolve(false, None, None, String::new())
            .unwrap()
            .is_none());
    }

    #[test]
    fn enabled_requires_repo() {
        // Thiếu repo → luôn lỗi (token có thể đến từ env nên không test riêng).
        assert!(HubConfig::resolve(true, None, Some("t".into()), String::new()).is_err());
    }

    #[test]
    fn enabled_with_repo_and_token_ok() {
        let cfg = HubConfig::resolve(true, Some("a/b".into()), Some("t".into()), "run1".into())
            .unwrap()
            .unwrap();
        assert_eq!(cfg.repo, "a/b");
        assert_eq!(
            cfg.repo_path(Path::new("/tmp/adapter.bpk")).unwrap(),
            "run1/adapter.bpk"
        );
        assert_eq!(
            HubConfig::resolve(true, Some("a/b".into()), Some("t".into()), String::new())
                .unwrap()
                .unwrap()
                .repo_path(Path::new("x.bpk"))
                .unwrap(),
            "x.bpk"
        );
    }
}
