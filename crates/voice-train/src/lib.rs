//! `voice-train` — huấn luyện + kiểm định cho đường voice conversion 1→1.
//!
//! Tách khỏi `voice-lib` (thư viện cho bên ngoài dùng) để crate chia sẻ không bao
//! giờ kéo theo phụ thuộc huấn luyện. Xem `docs/architecture.md` §13.6, §15.
//!
//! - [`mel`] — log-mel **khả vi** bằng Burn + loss `L1` (đường huấn luyện).
//! - [`data`] — nạp wav 16 kHz và features HuBERT `.npy` đã xuất.

pub mod data;
pub mod mel;
