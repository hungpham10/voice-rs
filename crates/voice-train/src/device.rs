//! Chọn thiết bị tính toán từ chuỗi CLI (`--device`) — §13.6.4e.
//!
//! Bản build hiện tại chỉ bật backend `flex` (CPU). `cuda`/`metal` **không** phải
//! lỗi cấu hình của người dùng mà là build chưa kèm feature — thông báo phải nói
//! rõ điều đó để không ai loay hoay vì lỗi tối nghĩa.

use burn::tensor::Device;

/// `cpu` (mặc định) hoặc báo lỗi rõ cho `cuda`/`metal`.
pub fn select(name: &str) -> Result<Device, String> {
    match name.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::default()),
        "cuda" | "gpu" => Err(
            "backend `cuda` chưa được build. Build lại với `--features cuda` (cần GPU NVIDIA) — xem §13.6.4e."
                .to_string(),
        ),
        "metal" => Err(
            "backend `metal` chưa được build. Build lại với `--features metal` (macOS) — xem §13.6.4e."
                .to_string(),
        ),
        other => Err(format!(
            "`--device` không hợp lệ: `{other}` (chọn: cpu, cuda, metal)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_ok() {
        assert!(select("cpu").is_ok());
        assert!(select("CPU").is_ok());
    }

    #[test]
    fn gpu_backends_report_build_feature() {
        assert!(select("cuda").unwrap_err().contains("cuda"));
        assert!(select("metal").unwrap_err().contains("metal"));
    }

    #[test]
    fn unknown_is_rejected() {
        assert!(select("tpu").unwrap_err().contains("không hợp lệ"));
    }
}
