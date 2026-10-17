//! Hợp đồng CSV giữa khâu chuẩn bị dữ liệu (Python, `scripts/data/`) và khâu
//! huấn luyện (Rust). Xem `docs/architecture.md` §13.6.4d.
//!
//! Định dạng cố định, một dòng một câu:
//!
//! ```text
//! split,speaker_id,utt_id,wav_path,feats_path,duration
//! ```
//!
//! Rust **chỉ** đọc định dạng này. Thêm dataset mới = thêm file Python bắn ra
//! cùng cột, **không** sửa Rust. Test `header_must_match` là hợp đồng: đổi cột ở
//! Python mà quên Rust thì CI bắt ngay.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Header bắt buộc — phải khớp **từng ký tự** (thứ tự cột là hợp đồng).
pub const HEADER: &str = "split,speaker_id,utt_id,wav_path,feats_path,duration";

/// Một câu trong manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    /// `train` hoặc `val` (giữ dạng chuỗi để mở rộng sau này).
    pub split: String,
    /// Định danh người nói — dùng để cân bằng batch.
    pub speaker_id: String,
    /// Định danh câu — dùng để đặt tên file features.
    pub utt_id: String,
    pub wav_path: String,
    pub feats_path: String,
    pub duration: f64,
}

impl Entry {
    /// Đã có features `.npy` chưa (đường dẫn tương đối tính từ thư mục hiện tại
    /// hoặc tuyệt đối — Python ghi sao dùng vậy).
    pub fn has_features(&self) -> bool {
        Path::new(&self.feats_path).exists()
    }

    pub fn has_wav(&self) -> bool {
        Path::new(&self.wav_path).exists()
    }
}

/// Đọc manifest từ chuỗi/reader. Bỏ dòng trống; header phải khớp [`HEADER`].
pub fn parse<R: BufRead>(reader: R) -> Result<Vec<Entry>, String> {
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| "manifest rỗng".to_string())?
        .map_err(|e| e.to_string())?;
    if header.trim() != HEADER {
        return Err(format!(
            "header manifest sai.\n  mong đợi: {HEADER}\n  nhận được: {}",
            header.trim()
        ));
    }

    let mut out = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let lineno = i + 2; // +2: 1-based và đã bỏ header
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.len() != 6 {
            return Err(format!(
                "dòng {lineno}: cần 6 cột, nhận {} — `{line}`",
                cols.len()
            ));
        }
        let duration: f64 = cols[5]
            .parse()
            .map_err(|_| format!("dòng {lineno}: duration không phải số — `{}`", cols[5]))?;
        if !duration.is_finite() || duration <= 0.0 {
            return Err(format!("dòng {lineno}: duration phải > 0 — `{duration}`"));
        }
        out.push(Entry {
            split: cols[0].to_string(),
            speaker_id: cols[1].to_string(),
            utt_id: cols[2].to_string(),
            wav_path: cols[3].to_string(),
            feats_path: cols[4].to_string(),
            duration,
        });
    }
    Ok(out)
}

/// Đọc manifest từ file.
pub fn read(path: &Path) -> Result<Vec<Entry>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("mở manifest {} lỗi: {e}", path.display()))?;
    parse(BufReader::new(file))
}

/// Ghi manifest ra file (dùng cho test và cho công cụ khác).
pub fn write(path: &Path, entries: &[Entry]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    writeln!(f, "{HEADER}").map_err(|e| e.to_string())?;
    for e in entries {
        writeln!(
            f,
            "{},{},{},{},{},{}",
            e.split, e.speaker_id, e.utt_id, e.wav_path, e.feats_path, e.duration
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Lọc theo `split` (ví dụ `"train"`, `"val"`). So khớp không phân biệt hoa thường.
pub fn filter_split(entries: &[Entry], split: &str) -> Vec<Entry> {
    entries
        .iter()
        .filter(|e| e.split.eq_ignore_ascii_case(split))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = "split,speaker_id,utt_id,wav_path,feats_path,duration\n\
train,spk01,u001,a.wav,f/u001.npy,3.5\n\
val,spk01,u002,b.wav,f/u002.npy,2.0\n";

    #[test]
    fn parses_valid_manifest() {
        let e = parse(Cursor::new(SAMPLE)).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].speaker_id, "spk01");
        assert_eq!(e[0].split, "train");
        assert_eq!(e[1].duration, 2.0);
    }

    /// Hợp đồng: đổi header phải fail rõ ràng, không parse mù.
    #[test]
    fn header_must_match() {
        let bad = "speaker,utt,wav,dur\ntrain,spk,u,a.wav,3.0\n";
        let err = parse(Cursor::new(bad)).unwrap_err();
        assert!(err.contains("header"), "err: {err}");
    }

    #[test]
    fn rejects_wrong_column_count() {
        let bad = format!("{HEADER}\ntrain,spk01,u001\n");
        assert!(parse(Cursor::new(bad)).unwrap_err().contains("6 cột"));
    }

    #[test]
    fn rejects_bad_duration() {
        let bad = format!("{HEADER}\ntrain,spk01,u001,a.wav,f.npy,abc\n");
        assert!(parse(Cursor::new(bad)).unwrap_err().contains("duration"));
        let neg = format!("{HEADER}\ntrain,spk01,u001,a.wav,f.npy,-1\n");
        assert!(parse(Cursor::new(neg)).unwrap_err().contains("> 0"));
    }

    #[test]
    fn skips_blank_lines_and_filters_split() {
        let s = format!("{HEADER}\n\n train, a,u1,x,y,1.0 \n\nval,b,u2,x,y,2.0\n");
        let e = parse(Cursor::new(s)).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(filter_split(&e, "train").len(), 1);
        assert_eq!(filter_split(&e, "VAL").len(), 1);
    }

    #[test]
    fn write_then_read_round_trip() {
        let entries = parse(Cursor::new(SAMPLE)).unwrap();
        let dir = std::env::temp_dir().join("voice_rs_manifest_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.csv");
        write(&path, &entries).unwrap();
        let back = read(&path).unwrap();
        assert_eq!(entries, back);
        let _ = std::fs::remove_file(&path);
    }
}
