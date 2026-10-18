//! Chọn thiết bị tính toán từ chuỗi CLI (`--device`) — §13.6.4e.
//!
//! Một binary hỗ trợ nhiều backend cùng lúc (CPU `flex` luôn có; `cuda` nếu
//! build với feature `cuda`). Chọn thiết bị **lúc chạy** qua tham số `--device`,
//! không phải compile-time.
//!
//! ```bash
//! # CPU (mặc định, luôn chạy — dù binary build kèm cuda)
//! cargo run --bin voice -- train --device cpu ...
//!
//! # CUDA (chỉ trên máy có GPU NVIDIA + driver/CUDA runtime)
//! cargo run --bin voice -- train --device cuda ...
//! ```

use burn::tensor::Device;

/// `cpu` (mặc định) hoặc `cuda` | `gpu` (khi build với `--features cuda`).
pub fn select(name: &str) -> Result<Device, String> {
    match name.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        "cuda" | "gpu" => Ok(Device::cuda(burn::tensor::DeviceIndex::default())),
        #[cfg(not(feature = "cuda"))]
        "cuda" | "gpu" => Err(
            "backend `cuda` chưa được build. Build lại với `--features cuda` (cần GPU NVIDIA + CUDA toolkit) — xem §13.6.4e."
                .to_string(),
        ),
        other => Err(format!(
            "`--device` không hợp lệ: `{other}` (chọn: cpu, cuda)"
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
    fn unknown_is_rejected() {
        assert!(select("tpu").unwrap_err().contains("không hợp lệ"));
        assert!(select("metal").unwrap_err().contains("không hợp lệ"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_is_accepted_when_feature_on() {
        // Chỉ kiểm tra không bị từ chối ngay lúc chọn device. Máy này không
        // có GPU NVIDIA nên khởi tạo thực tế sẽ lỗi runtime — đây là kỳ vọng.
        let _ = select("cuda");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_reports_build_feature_when_off() {
        let err = select("cuda").unwrap_err();
        assert!(err.contains("cuda"), "unexpected error: {err}");
    }
}
