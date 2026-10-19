//! `voice-train` — công cụ CLI `voice`: huấn luyện + kiểm định cho đường voice conversion 1→1.
//!
//! Tách khỏi `voice-lib` (thư viện cho bên ngoài dùng) để crate chia sẻ không bao
//! giờ kéo theo phụ thuộc huấn luyện. Xem `docs/architecture.md` §13.6, §15.
//!
//! - [`mel`] — log-mel **khả vi** bằng Burn + loss `L1` (đường huấn luyện).
//! - [`data`] — nạp wav 16 kHz, features `.npy`, ghi wav.
//! - [`manifest`] — hợp đồng CSV giữa khâu dữ liệu (Python) và huấn luyện (Rust).
//! - [`dataset`] — dataset nhiều giọng + cắt đoạn + RNG xác định.
//! - [`device`] — chọn backend từ `--device`.
//! - [`train`] — `voice train` (Luồng 1, đường huấn luyện duy nhất).
//! - [`render`] — `render`: suy diễn feats + adapter → wav (không huấn luyện).
//! - [`eval`] — chỉ số + cổng PASS/FAIL.
//! - [`diag`] — chẩn đoán rè/nhiễu: mel-L1 theo band + voiced/unvoiced (PR-B1).
//! - [`hub`] — đẩy kết quả lên HuggingFace Hub (mặc định tắt).

pub mod data;
pub mod dataset;
pub mod device;
pub mod diag;
pub mod eval;
pub mod hub;
pub mod manifest;
pub mod mel;
pub mod render;
pub mod streaming;
pub mod train;
