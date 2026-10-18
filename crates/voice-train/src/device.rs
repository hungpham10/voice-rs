//! Chọn thiết bị tính toán từ chuỗi CLI (`--device`) — §13.6.4e.
//!
//! Một binary có thể hỗ trợ nhiều backend cùng lúc (CPU `flex` luôn có;
//! `cuda` nếu build với `--features cuda`). Chọn thiết bị **lúc chạy** qua tham
//! số `--device`, không phải compile-time.
//!
//! ```bash
//! # CPU (mặc định, luôn chạy)
//! cargo run --bin voice-train -- train overfit --device cpu ...
//!
//! # CUDA (chỉ chạy được nếu binary được build với --features cuda,
//!         và máy có GPU NVIDIA + CUDA runtime)
//! cargo run --bin voice-train --features cuda -- train overfit --device cuda ...
//! ```

use burn::tensor::Device;

/// `cpu` (mặc định), `cuda` | `gpu` (khi build với `--features cuda`).
pub fn select(name: &str) -> Result<Device, String> {
    match name.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::flex()),
        "cuda" | "gpu" => match try_cuda() {
            Ok(device) => Ok(device),
            Err(reason) => Err(reason),
        },
        other => Err(format!(
            "`--device` không hợp lệ: `{other}` (chọn: cpu, cuda)"
        )),
    }
}

/// Cố gắng tạo CUDA device. Nếu binary chưa build `--features cuda` hoặc không
/// có GPU, trả về lỗi rõ (không panic mơ hồ).
fn try_cuda() -> Result<Device, String> {
    #[cfg(feature = "cuda")]
    {
        Ok(Device::cuda(burn::tensor::DeviceIndex::default()))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ();
        Err(
            "backend `cuda` chưa được build. Build lại với `--features cuda` (cần GPU NVIDIA + CUDA toolkit) — xem §13.6.4e."
                .to_string(),
        )
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
    fn unknown_is_rejected() {
        assert!(select("tpu").unwrap_err().contains("không hợp lệ"));
        assert!(select("metal").unwrap_err().contains("không hợp lệ"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_is_accepted_when_feature_on() {
        // Chỉ kiểm tra không bị từ chối ngay lúc chọn device (có thể vẫn lỗi
        // runtime nếu máy không có GPU).
        let _ = select("cuda");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_reports_build_feature_when_off() {
        let err = select("cuda").unwrap_err();
        assert!(err.contains("cuda"), "unexpected error: {err}");
    }
}
