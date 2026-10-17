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

GitHub Release  v2.2.10  (cargo-dist)  ──►  voice-{hubert,train,models}-....tar.xz
   • embed  ·  voice-train  ·  verify_dec   (Linux x86_64)

repo  samples/422-122949-0031.wav     ←   wav mẫu (nằm trong git)
```

Ba binary:

| Binary | Việc |
|---|---|
| `embed` | wav giọng đích → features ContentVec `[1,T,768]` (trọng số nhúng sẵn trong binary) |
| `verify_dec` | validate `dec` (Rust) khớp PyTorch (bước kiểm tra trước khi train) |
| `voice-train` | công cụ huấn luyện thống nhất: `train overfit` (1 câu), `train bridge` (nhiều giọng), `eval` |

---

## 3. Cách nhanh — dùng notebook

1. Mở [`notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb) (hoặc upload
   lên Colab).
2. **Runtime → Change runtime type → CPU**.
3. Chạy lần lượt 4 cell:
   - **Cell 1** — clone repo.
   - **Cell 2** — `./scripts/fetch_assets.sh assets-v1` (tải model, verify sha256).
   - **Cell 3** — `./scripts/fetch_bin.sh` (tải binary → `./bin/`).
   - **Cell 4** — `./scripts/run_training.sh` (validation → training).
4. Cell 5 tải `test_output.wav` + `adapter.bpk` về máy.

---

## 4. Cách thủ công (CLI)

Trong một cell Colab (hoặc terminal):

```bash
cd /content
git clone https://github.com/hungpham10/voice-rs.git
cd voice-rs

# 1) model (ContentVec ONNX + dec) — verify SHA256SUMS
./scripts/fetch_assets.sh                 # tag mặc định: assets-v1
# ./scripts/fetch_assets.sh --f0          # thêm rmvpe_f0.onnx nếu cần voice-f0

# 2) binary train/validate (Linux x86_64) → ./bin/
./scripts/fetch_bin.sh                    # tag mặc định: v2.2.10

# 3) validation + training
SEG=86 STEPS=250 LR=1e-3 ./scripts/run_training.sh
```

`run_training.sh` sẽ tự:
1. sinh features nếu còn thiếu (`embed`),
2. chạy **validation** (`verify_dec`) — so `dec` với ground truth PyTorch,
3. chạy **training** (`voice-train train overfit`) — ghi `test_output.wav` + `sources/features/adapter.bpk`.

Nếu `./bin/` chưa có binary, script tự rơi về `cargo run` (dành cho máy dev có Rust).

### Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SEG` | cả câu (tự co theo dữ liệu) | số frame latent đem ra huấn luyện; `86` ≈ 1 giây |
| `STEPS` | `300` | số bước tối ưu |
| `LR` | `1e-3` | learning rate |
| `WAV` | `sources/422-122949-0031.wav` | wav giọng đích |
| `FEATS` | `sources/features/422-122949-0031.npy` | features của wav đích |
| `DEC_DIR` | `sources/vits_ljs_dec` | thư mục trọng số `dec` |
| `OUT_WAV` | `test_output.wav` | wav sinh ra sau train |
| `OUT_ADAPTER` | `sources/features/adapter.bpk` | checkpoint adapter |

Ví dụ smoke 1 giây:

```bash
SEG=86 STEPS=40 LR=1e-3 ./scripts/run_training.sh
```

---

## 4b. Nhiều giọng — Luồng 1 (`train bridge`)

Khác với `run_training.sh` (overfit **một** câu), Luồng 1 học cầu nối nội dung trên
**nhiều giọng** và chấm điểm bằng cổng PASS/FAIL (§13.6.4). Cần dataset đã chia
thành `train.csv`/`val.csv` bằng `scripts/data/` (Python — xem
`scripts/data/README.md`).

Trên Colab, nếu `./bin/` có `embed_batch` + `voice-train` (từ `fetch_bin.sh`) thì
không cần Rust:

```bash
# dữ liệu đã copy lên /content/data/vivos (train.csv, val.csv, wav/, …)
DATA=/content/data/vivos STEPS=1500 SEG=86 BATCH=8 ./scripts/run_bridge_training.sh
```

`run_bridge_training.sh` tự: sinh features (`embed_batch`) → `train bridge` →
`eval` (thoát **mã 1** nếu FAIL). Kết quả ở `$DATA/bridge/`: `adapter_best.bpk`,
`adapter.bpk`, `history.csv`, `report.csv`, `summary.json`.

Đọc chỉ số `eval`: `mel-L1` (càng nhỏ càng tốt), `speaker_gap` (nhỏ = công bằng giữa
giọng), `drift` (phải ≈ 0), RMS (bắt audio im lặng/nổ). Để so giữa các version, giữ
`summary.json` của version trước rồi chạy eval với `--baseline summary.json`
(`--max-regression 0.05`) — thoái lui > 5% sẽ FAIL. Xem `docs/architecture.md`
§13.6.4g.

---

## 5. Dùng giọng đích của bạn

`train overfit` cần **hai** thứ khớp nhau:

1. một file **wav mono** (khuyến nghị 16 kHz, PCM) — giọng đích,
2. **features ContentVec** của chính wav đó (`[1,T,768]`, do `embed` sinh).

Cách làm:

```bash
# giả sử bạn có my_voice.wav (16 kHz mono)
mkdir -p sources/features

./bin/embed my_voice.wav sources/features/my_voice.npy      # sinh features

WAV=my_voice.wav \
FEATS=sources/features/my_voice.npy \
OUT_WAV=my_out.wav \
SEG=86 STEPS=250 LR=1e-3 \
./scripts/run_training.sh
```

Ghi chú:
- `SEG` được **tự động co lại** theo độ dài wav + features thật, nên audio ngắn
  không gây lỗi. Muốn dùng trọn câu thì để trống `SEG` (mặc định cả câu).
- Nếu wav không phải 16 kHz, chương trình vẫn chạy (resample tuyến tính về
  22.05 kHz) nhưng chất lượng tốt nhất khi đầu vào là 16 kHz.
- Training hiện **overfit một speaker trên một câu** — đây là bước chứng minh
  adapter học được, chưa phải huấn luyện nhiều câu.

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
| `không thấy bin/embed — dùng cargo run` | chưa chạy `fetch_bin.sh`; nếu máy có Rust thì script tự build |
| Lỗi `Exec format error` | binary là Linux x86_64 — phải chạy trên Colab/Linux, không chạy trên macOS |
| Hết RAM | `embed` cầm ~360 MB trọng số; giảm `SEG`, đóng runtime khác |
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
