# voice-rs

Hệ **voice conversion 1→1** (một giọng nguồn → một giọng đích) viết bằng **Rust /
[Burn](https://github.com/tracel-ai/burn)**, chạy suy luận **CPU-first**, kèm phụ đề
(closed caption) neo theo mốc thời gian của video nguồn.

> Inference hoàn toàn bằng Rust. Trọng số và ground truth đã được export **một lần**
> rồi đóng băng trên HuggingFace — repo **không có Python trong pipeline/runtime**
> (chỉ còn 1 công cụ dev `scripts/screen_onnx.py`, chạy tay khi nạp ONNX mới).

Tài liệu kiến trúc đầy đủ (nguồn sự thật duy nhất): **[`docs/architecture.md`](docs/architecture.md)**.

## Trạng thái

| Crate | Trách nhiệm | Trạng thái |
|---|---|---|
| `voice-hubert` | ContentVec 768 @ 50 Hz | ✅ max abs diff `3e-4` |
| `voice-mel` | log-mel (nhiều variant) | ✅ `4.77e-07` |
| `voice-f0` | RMVPE, f0 @ 100 Hz | ✅ mean diff `1.78 Hz` |
| `voice-models` | VITS HiFi-GAN generator (`dec`), viết tay | ✅ `1.26e-08` vs PyTorch |
| `voice-lib` | Thư viện dùng chung: `cc` (dấu câu, frame, `.srt`) + `vc` (`Adapter`, resample) | 🟡 43 unit test |
| `voice-train` | Huấn luyện/validate adapter (log-mel khả vi bằng Burn) | 🟡 loss giảm `2.22×`/40 bước |

## Cấu trúc

```
crates/
├── voice-hubert/   ContentVec embedder (ONNX) — 768 @ 50 Hz
├── voice-mel/      log-mel variants (rosa-based)
├── voice-f0/       RMVPE f0 @ 100 Hz (ONNX)
├── voice-models/   VITS HiFi-GAN `dec` viết tay bằng Burn
├── voice-lib/      thư viện dùng chung (cho bên ngoài)
└── voice-train/    huấn luyện + kiểm định (không kéo vào voice-lib)
docs/architecture.md   nguồn sự thật
notebooks/colab_train.ipynb   huấn luyện adapter trên Colab (CPU)
samples/               audio test nhỏ, commit được
scripts/               nạp model/binary + chạy huấn luyện (Rust/shell; còn `screen_onnx.py` là công cụ dev)
```

## Chạy nhanh (local)

Cần Rust stable. Trọng số tải riêng (xem dưới).

```bash
./scripts/setup.sh --no-bin    # MỘT lệnh: model (sources/) + phụ thuộc Python (máy dev bỏ qua binary)
cargo test -p voice-lib -p voice-mel -p voice-models -p voice-train   # CI chạy đúng bộ này
./scripts/run_training.sh      # validation (verify_dec) → training adapter (tự build nếu chưa có binary)
```

> Trên Colab (không có Rust) dùng `./scripts/setup.sh` — tự tải thêm binary Linux
> vào `./bin/`. Xem [`docs/colab-guide.md`](docs/colab-guide.md).

## Model (không nằm trong git)

`sources/` bị `.gitignore`. Cách gọn nhất: `./scripts/fetch_assets.sh` tải bundle
**chỉ-model** từ Release [`assets-v1`](https://github.com/hungpham10/voice-rs/releases/tag/assets-v1):

| Bundle | Chứa |
|---|---|
| `voice-assets.tar.gz` | ContentVec ONNX + `vits_ljs_dec` (VITS generator đã export) |
| `f0-assets.tar.gz` (`--f0`) | `rmvpe_f0.onnx` (cho `voice-f0`) |

**Không cần PyTorch** trên máy đích (export đã làm trong CI). Bundle kèm `SHA256SUMS`
để `fetch_assets.sh` tự verify. Dữ liệu đích (dataset) tự tải khi cần.

Nguồn pull gốc cho workflow nằm ở [`scripts/asset_checksums.txt`](scripts/asset_checksums.txt):
tất cả model (ContentVec ONNX, `rmvpe_f0.onnx`, `vits_ljs_dec/`) nằm trên HF của mình —
[`hung0913208/voice-rs-assets`](https://huggingface.co/hung0913208/voice-rs-assets) — mỗi file
kèm sha256 để verify. Không còn `pretrained_ljs.pth` (`dec` đã export sẵn) nên **không cần
Python/PyTorch** ở bất kỳ bước nào.

Hoặc dùng thẳng notebook `notebooks/colab_train.ipynb` — nó gọi 2 script trên.

## Binary (cho Colab — không cần build)

Trên Colab **không cần cài Rust hay biên dịch**: binary cho Linux x86_64 được
[`cargo-dist`](https://opensource.axo.dev/cargo-dist/) build sẵn trong CI và đăng
lên GitHub Release khi push một tag phiên bản.

```bash
./scripts/setup.sh                  # một lệnh: model + phụ thuộc Python + binary → ./bin/
./scripts/run_training.sh           # dùng luôn binary, không đụng cargo
```

| Binary | Crate | Việc |
|---|---|---|
| `embed` | `voice-hubert` | wav → features `[1,T,768]` (trọng số ContentVec **nhúng trong binary**) |
| `voice-train` | `voice-train` | công cụ huấn luyện thống nhất: `train overfit`, `train bridge`, `eval` |
| `verify_dec` | `voice-models` | validate `dec` (Rust) vs PyTorch |

Các binary này **thuần Rust, tĩnh** — kiểm bằng báo cáo linkage của cargo-dist
(chỉ phụ thuộc `libc`/`libm`/`libgcc`, không OpenSSL hay thư viện ngoài).

📖 Hướng dẫn chạy trên Colab từng bước: **[`docs/colab-guide.md`](docs/colab-guide.md)**.

## Giấy phép

MIT — xem [`LICENSE`](LICENSE).

Repo port kiến trúc VITS HiFi-GAN generator sang Rust/Burn.
`samples/422-122949-0031.wav` trích từ **LibriSpeech** (`test-clean`, reader 422),
giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
