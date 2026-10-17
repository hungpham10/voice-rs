//! `voice-train` — huấn luyện + kiểm định cho đường voice conversion 1→1.
//!
//! Tách khỏi `voice-lib` (thư viện cho bên ngoài dùng) để crate chia sẻ không bao
//! giờ kéo theo phụ thuộc huấn luyện. Xem `docs/architecture.md` §13.6, §15.
//!
//! - [`mel`] — log-mel **khả vi** bằng Burn + loss `L1` (đường huấn luyện).
//! - [`data`] — nạp wav 16 kHz, features `.npy`, ghi wav.
//! - [`manifest`] — hợp đồng CSV giữa khâu dữ liệu (Python) và huấn luyện (Rust).
//! - [`dataset`] — dataset nhiều giọng + cắt đoạn + RNG xác định.
//! - [`device`] — chọn backend từ `--device`.
//! - [`train`] — `train bridge` (Luồng 1).
//! - [`overfit`] — `train overfit` (GĐ4, 1 câu).
//! - [`eval`] — chỉ số + cổng PASS/FAIL.
//! - [`hub`] — đẩy kết quả lên HuggingFace Hub (mặc định tắt).

pub mod data;
pub mod dataset;
pub mod device;
pub mod eval;
pub mod hub;
pub mod manifest;
pub mod mel;
pub mod overfit;
pub mod train;
