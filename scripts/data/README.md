# Chuẩn bị dữ liệu (Python — dev tooling)

Khâu này **chỉ** tạo dữ liệu cho huấn luyện; **không** nằm trong runtime suy luận
(Rust). Mọi thứ dùng Python ở đây vì thư viện xử lý audio/dataset của Python chín
hơn — quyết định có chủ đích.

## Luồng

```
prepare.py  ──►  manifest_raw.csv        (speaker_id,utt_id,wav_path,duration)
                     │  (mọi wav đã 16 kHz mono, ở <out>/wav/)
   split.py   ──►  train.csv / val.csv   (hợp đồng 6 cột cho Rust)
                     │  (feats_path = <out>/features/<utt_id>.npy)
 voice embed  ──►  <out>/features/*.npy
 voice         ──► train → validate (cổng PASS/FAIL)
```

## Cài

```bash
pip install -r scripts/data/requirements.txt
```

## 1) Chuẩn bị một dataset

```bash
# VIVOS (16 kHz). Cần dữ liệu đã giải nén (--src), hoặc tải sẵn từ mirror HF (--download).
python scripts/data/prepare.py vivos       --src /path/VIVOS  --out data/vivos
python scripts/data/prepare.py vivos --download --out data/vivos   # tải data/vivos.tar.gz (~1.5 GB) rồi giải nén
# Common Voice vi (clips/ + validated.tsv) — mp3 được resample về 16 kHz.
python scripts/data/prepare.py common-voice --src /path/cv_vi --out data/cv
# viVoice — thư mục <speaker>/*.wav
python scripts/data/prepare.py vivvoice     --src /path/viVoice --out data/vivvoice

# thử nhanh: --limit-per-speaker 20
```

Ba adapter ra **cùng một** `manifest_raw.csv` → thêm dataset mới = thêm một
subcommand, **không** đụng Rust.

## 2) Chia train/val

```bash
python scripts/data/split.py --manifest data/vivos/manifest_raw.csv \
    --out-dir data/vivos --val-ratio 0.1
```

- Chia **theo từng câu, trong từng giọng**, tất định bằng `sha256(utt_id)`.
- Bảo đảm mỗi giọng có ≥1 câu train và ≥1 câu val (giọng lẻ để train).
- Ghi `train.csv` + `val.csv` đúng hợp đồng cột.

> Hợp đồng 6 cột `split,speaker_id,utt_id,wav_path,feats_path,duration` được Rust
> kiểm bằng test `manifest::header_must_match`. Đổi cột ở đây mà quên Rust sẽ đỏ CI.

## 3) Sinh features + huấn luyện

Xem `scripts/run_training.sh` (gộp cả 3 bước), hoặc chạy tay:

```bash
cargo run --release -p voice-train --bin voice -- embed --manifest data/vivos/train.csv --skip-existing
cargo run --release -p voice-train --bin voice -- embed --manifest data/vivos/val.csv   --skip-existing
cargo run --release -p voice-train --bin voice -- train \
  --train-manifest data/vivos/train.csv --val-manifest data/vivos/val.csv --out-dir data/vivos/bridge
cargo run --release -p voice-train --bin voice -- validate \
  --val-manifest data/vivos/val.csv --adapter data/vivos/bridge/adapter_best.bpk
```


## Test (tuỳ chọn)

```bash
python -m pytest scripts/data/test_split.py
```
