//! Chọn thiết bị tính toán từ chuỗi CLI (`--device`) — §13.6.4e.
//!
//! Mặc định chỉ bật backend `flex` (CPU). Bật feature `cuda` lúc compile
//!
//! ```bash
//! cargo build --release -p voice-train --features cuda  # cần CUDA toolkit
//! ```
//! thì `--device cuda` sẽ hoạt động. Trên macOS có thể bật `metal` tương tự.

use burn::tensor::Device;

/// `cpu` (mặc định), `cuda` (khi bật feature), `metal` (khi bật feature).
pub fn select(name: &str) -> Result<Device, String> {
    match name.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::default()),
        #[cfg(feature = "cuda")]
        "cuda" | "gpu" => Ok(Device::cuda(burn::tensor::DeviceIndex::default())),
        #[cfg(not(feature = "cuda"))]
        "cuda" | "gpu" => Err(
            "backend `cuda` chưa được build. Build lại với `--features cuda` (cần GPU NVIDIA + CUDA toolkit) — xem §13.6.4e."
                .to_string(),
        ),
        #[cfg(feature = "metal")]
        "metal" => Ok(Device::metal(burn::tensor::DeviceKind::default())),
        #[cfg(not(feature = "metal"))]
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
        #[cfg(feature = "cuda")]
        {
            // Khi bật feature, cuda được chấp nhận (có thể vẫn lỗi runtime nếu không có GPU)
            let _ = select("cuda");
        }
        #[cfg(not(feature = "cuda"))]
        {
            assert!(select("cuda").unwrap_err().contains("cuda"));
        }
        #[cfg(feature = "metal")]
        {
            let _ = select("metal");
        }
        #[cfg(not(feature = "metal"))]
        {
            assert!(select("metal").unwrap_err().contains("metal"));
        }
    }

    #[test]
    fn unknown_is_rejected() {
        assert!(select("tpu").unwrap_err().contains("không hợp lệ"));
    }
}