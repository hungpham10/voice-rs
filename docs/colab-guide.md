# Chạy trên Colab — hướng dẫn huấn luyện adapter

Tài liệu này hướng dẫn **chạy huấn luyện adapter (GĐ4) trên Colab CPU** mà
**không cần cài Rust và không biên dịch gì**: Colab chỉ tải model + binary đã
build sẵn rồi chạy.

> Kiến trúc/chi tiết kỹ thuật: [`architecture.md`](architecture.md). Tài liệu này
> chỉ nói *cách chạy*.

---

## 1. Bạn cần gì (và không cần gì)

| Cần | Không cần |
|---|---|
| Tài khoản Google (Colab) | Cài Rust / `cargo` |
| Repo public `hungpham10/voice-rs` | PyTorch |
| Chọn runtime **CPU** (chưa hỗ trợ GPU) | GPU (chưa nằm trong phạm vi) |

Binary đã build sẵn cho **Linux x86_64** (đúng môi trường Colab), nên chỉ tải về
chạy.

---

## 2. Các artifact sẽ tải

Hai nguồn, tải tự động bằng script:

```
HuggingFace hung0913208/voice-rs-assets
        │  (CI release-assets.yml đóng gói)
        ▼
GitHub Release  assets-v1            ──►  voice-assets.tar.gz
   • sources/contentvec/contentvec_768l12.onnx
   • sources/vits_ljs_dec/            (155 .npy + config + ref_*.npz)

GitHub Release  v2.2.11  (cargo-dist)  ──►  voice-{hubert,train,models}-....tar.xz
   • voice   (Linux x86_64)

repo  samples/422-122949-0031.wav     ←   wav mẫu (nằm trong git)
```

Hai binary (build cho Colab):

| Binary | Việc |
|---|---|
| `voice` | một lệnh thống nhất: `embed` / `train` / `validate` / `generate` / `diag` / `verify` — trọng số ContentVec nhúng sẵn trong binary |

> **GPU (tùy chọn — release đã có sẵn):**
> Binary `voice` trên Release đã build kèm feature `cuda`.
> Một binary duy nhất cho Colab và máy có GPU:
> ```bash
> # CPU (mặc định — Colab)
> ./voice train --device cpu ...
> # CUDA (máy có NVIDIA GPU + driver/CUDA runtime)
> ./voice train --device cuda ...
> ```
> Binary GPU phụ thuộc thư viện hệ thống (`libcudart` qua `dlopen` lúc runtime) —
> `--device cpu` vẫn chạy bình thường trên máy không có NVIDIA.

---

## 3. Cách nhanh — dùng notebook

1. Mở [`notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb) (hoặc upload
   lên Colab).
2. **Runtime → Change runtime type → CPU**.
3. Chạy lần lượt 4 cell:
   - **Cell 1** — clone repo.
   - **Cell 2** — `./scripts/fetch_assets.sh assets-v1` (tải model, verify sha256).
   - **Cell 3** — `./scripts/fetch_bin.sh` (tải binary → `./bin/`).
   - **Cell 4** — `run_training.sh` (embed → train → validate).

---

## 4. Cách thủ công (CLI)

Trong một cell Colab (hoặc terminal):

```bash
cd /content
git clone https://github.com/hungpham10/voice-rs.git
cd voice-rs

# 1) CÀI MỘT LỆNH: model (sources/) + binary (bin/) + phụ thuộc Python
./scripts/setup.sh

# —— hoặc làm từng bước ——
# ./scripts/fetch_assets.sh                 # model, verify SHA256SUMS (tag assets-v1)
# ./scripts/fetch_assets.sh --f0            # thêm rmvpe_f0.onnx nếu cần voice-f0
# ./scripts/fetch_bin.sh                    # binary Linux x86_64 → ./bin/

# 2) sinh features + training + validate (một lệnh)
DATA=data/my ./scripts/run_training.sh
```

`run_training.sh` sẽ tự:
1. sinh features cho cả manifest còn thiếu (`voice embed --skip-existing`),
2. chạy **training** (`voice train`) — ghi adapter + history vào `--out-dir`,
3. chạy **validate** (`voice validate`) — cổng PASS/FAIL (thoát **mã 1** nếu FAIL).

Nếu `./bin/` chưa có binary, script tự rơi về `cargo run` (dành cho máy dev có Rust).

### Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DATA` | `data/vivos` | thư mục dataset (chứa manifest + out-dir) |
| `TRAIN_MANIFEST` / `VAL_MANIFEST` | `$DATA/train.csv` / `$DATA/val.csv` | manifest CSV hợp đồng |
| `OUT_DIR` | `$DATA/bridge` | nơi ghi adapter/history/report |
| `SEG` | `86` | frame latent mỗi mẫu huấn luyện |
| `BATCH` | `8` | số câu mỗi bước |
| `STEPS` | `1500` | số bước tối ưu |
| `LR` | `1e-3` | learning rate |
| `DEVICE` | `cpu` | `cpu`/`cuda` |
| `BIN_DIR` | `bin` | thư mục binary đã tải; thiếu thì `cargo run` |

```bash
# VIVOS đầy đủ:
DATA=data/vivos STEPS=1500 ./scripts/run_training.sh
```

### Nghe lại cả câu từ adapter đã train (không train lại)
`Adapter` là `Linear` không phụ thuộc thời gian → dùng `generate` để sinh **cả câu**:
```bash
./bin/voice generate \
  --feats sources/features/422-122949-0031.npy \
  --adapter sources/features/adapter.bpk \
  --out-wav full.wav            # 6.84 s, không đụng weight
```

---

## 4b. Nhiều giọng — Luồng 1 (`voice train`)

Luồng 1 học cầu nối nội dung trên
**nhiều giọng** và chấm điểm bằng cổng PASS/FAIL (§13.6.4). Cần dataset đã chia
thành `train.csv`/`val.csv` bằng `scripts/data/` (Python — xem
`scripts/data/README.md`).

Trên Colab, nếu `./bin/` có `voice` (từ `fetch_bin.sh`) thì
không cần Rust:

```bash
# dữ liệu đã copy lên /content/data/vivos (train.csv, val.csv, wav/, …)
DATA=/content/data/vivos STEPS=1500 SEG=86 BATCH=8 ./scripts/run_training.sh
```

`run_training.sh` tự: sinh features (`voice embed`) → `voice train` →
`eval` (thoát **mã 1** nếu FAIL). Kết quả ở `$DATA/bridge/`: `adapter_best.bpk`,
`adapter.bpk`, `history.csv`, `report.csv`, `summary.json`.

Đọc chỉ số `eval`: `mel-L1` (càng nhỏ càng tốt), `speaker_gap` (nhỏ = công bằng giữa
giọng), `drift` (phải ≈ 0), RMS (bắt audio im lặng/nổ). Để so giữa các version, giữ
`summary.json` của version trước rồi chạy eval với `--baseline summary.json`
(`--max-regression 0.05`) — thoái lui > 5% sẽ FAIL. Xem `docs/architecture.md`
§13.6.4g.

---

## 5. Dùng giọng đích của bạn

`voice` cần **hai** thứ khớp nhau cho một câu:

1. một file **wav mono** (khuyến nghị 16 kHz, PCM) — giọng đích,
2. **features ContentVec** của chính wav đó (`[1,T,768]`, do `voice embed` sinh).

Cách làm:

```bash
# giả sử bạn có my_voice.wav (16 kHz mono)
mkdir -p sources/features

./bin/voice embed --wav my_voice.wav --out sources/features/my_voice.npy

# 1 dòng manifest (song song khâu dataset) rồi chạy script duy nhất
mkdir -p data/my
printf 'split,speaker_id,utt_id,wav_path,feats_path,duration\n' > data/my/manifest.csv
printf 'train,my,my_voice,my_voice.wav,sources/features/my_voice.npy,0\n' >> data/my/manifest.csv
DATA=data/my TRAIN_MANIFEST=data/my/manifest.csv VAL_MANIFEST=data/my/manifest.csv \
STEPS=250 LR=1e-3 ./scripts/run_training.sh
```

Ghi chú:
- Training hiện **nhiều câu từ manifest** (đường duy nhất); 1 dòng manifest =
  1 câu — dùng cho smoke. Câu ngắn không gây lỗi (kích thước đoạn tự co theo
  dữ liệu thật).
- Nếu wav không phải 16 kHz, chương trình vẫn chạy (resample tuyến tính về
  22.05 kHz) nhưng chất lượng tốt nhất khi đầu vào là 16 kHz.

---
## 6. Kết quả

| File | Nghĩa |
|---|---|
| `test_output.wav` | câu nói sinh ra bằng adapter đã học (22.05 kHz) |
| `sources/features/adapter.bpk` | trọng số adapter (`Linear 768→192`) đã lưu |

Tiêu chí **PASS**: `mel-L1` giảm rõ rệt qua các bước (dòng cuối `PASS — loss giảm N×`).

---

## 7. Thời gian & chi phí (Colab CPU)

Colab free ~2 core. `dec` forward+backward ở Flex CPU khá nặng:

| Cấu hình | Thời gian/bước | 250 bước |
|---|---|---|
| `SEG=86` (≈1 s audio) | ~20–30 s | ~1.5–2 giờ |
| cả câu (589 frame) | ~2–3 phút | ~8–12 giờ |

→ Nên bắt đầu bằng `SEG=86 STEPS=40` để chắc luồng chạy thông, rồi mới tăng.

---

## 8. Sự cố thường gặp

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| `✗ thiếu sources/vits_ljs_dec` | chưa chạy `./scripts/fetch_assets.sh` |
| `curl: 404` khi `fetch_bin.sh` | sai tag. Kiểm tra `TAG=vX.Y.Z` đúng release có binary; xem mục Releases |
| `sha256sum -c` FAIL khi fetch model | file tải lỗi mạng — chạy lại `fetch_assets.sh` |
| `không thấy bin/voice — dùng cargo run` | chưa chạy `fetch_bin.sh`; nếu máy có Rust thì script tự build |
| Lỗi `Exec format error` | binary là Linux x86_64 — phải chạy trên Colab/Linux, không chạy trên macOS |
| Hết RAM | `voice embed` cầm ~360 MB trọng số; giảm `BATCH`, đóng runtime khác |
| `loss KHÔNG giảm` | tăng `STEPS`, kiểm tra wav/features đúng cặp, thử `LR` khác |

---

## 9. Muốn chạy inference (không phải Colab)?

Phần **inference** là **library Rust** trong repo (các crate `voice-*`). Clone repo
và build cục bộ (cần Rust + `./scripts/fetch_assets.sh` để có ONNX cho bước codegen
lúc build). Binary trên Release chỉ phục vụ **train/validate trên Colab**.

---

## 10. Ghi chú kỹ thuật (vì sao không cần build)

- Binary được đóng gói bằng [`cargo-dist`](https://opensource.axo.dev/cargo-dist/)
  trong CI (`.github/workflows/release.yml`) khi push tag phiên bản.
- Chỉ target `x86_64-unknown-linux-gnu`; không installer, không kèm mã nguồn.
- Toàn bộ phụ thuộc là **thuần Rust tĩnh** — không OpenSSL/libz/libtorch; chỉ link
  `libc`/`libm`/`libgcc` (luôn có sẵn). `cargo-dist` in báo cáo linkage mỗi lần build.
- `embed` **nhúng trọng số ContentVec** vào binary (`VOICE_EMBED_WEIGHTS=1`), nên nó
  tự chứa, không cần file `.bpk` lúc chạy (đổi lại kích thước ~373 MB).
