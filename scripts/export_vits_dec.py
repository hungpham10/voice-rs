#!/usr/bin/env python3
"""Xuất trọng số VITS HiFi-GAN generator (`dec`) ra `.npy` từng tensor.

Vì sao tách riêng
-----------------
`dec` là phần duy nhất của VITS mà ta viết tay bằng Burn (§2 của
docs/architecture.md). Nó là conv thuần — 200 node ONNX, không `Expand`,
không ngẫu nhiên — nên không cần burn-onnx codegen.

Việc cần làm ở đây
------------------
1. Lọc các tensor có prefix `dec.` trong checkpoint.
2. **Decompose weight-norm**: checkpoint VITS cũ lưu `weight_g` / `weight_v`,
   chưa gộp. Ta gộp ngay ở Python để Rust nhận đúng `W`:
       W = g · v / ‖v‖        (norm lấy theo mọi chiều trừ chiều cuối)
3. Xuất ra `.npy`, tên file mirror y hệt key trong PyTorch.
4. Đồng thời chạy PyTorch để lấy ground truth cho nhiều `T`.

Kiểm tra ngay khi xuất
---------------------
- `conv_post.bias` phải **vắng mặt** (`bias=False`) — bẫy số 3 ở §2.3.
- 4 `ups` phải khớp `upsample_rates=[8,8,2,2]`,
  `upsample_kernel_sizes=[16,16,4,4]`, `upsample_initial_channel=512`.
- 12 ResBlock, mỗi cái 6 conv (`convs1` 3 + `convs2` 3).
- `convs2` dùng CÙNG kernel_size với `convs1` — bẫy số 2 ở §2.3.
- Chiều dài output đúng `T * 256` ở **mọi** T, kể cả T lẻ.
"""

import argparse
import json
import os

import numpy as np
import torch

CKPT = "sources/pretrained_ljs.pth"
HPS = "sources/vits/configs/ljs_base.json"
OUT = "sources/vits_ljs_dec"

# ---------------------------------------------------------------- weight-norm


def wn_compose(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Ghép weight-norm theo đúng công thức PyTorch: `W = v · g / ‖v‖`.

    PyTorch `weight_norm(dim=0)` chuẩn hóa trên **mọi chiều trừ chiều 0**: norm
    lấy trên `dims = (1, …, -1)`. Vì `g` có dạng `[out, 1, 1]` (đã kiểm: `ups.0`
    là `(512, 1, 1)` trên `weight_v (512, 256, 16)`), đây là norm **theo từng
    kênh ra**, KHÔNG phải "mọi chiều trừ chiều cuối" (đó là `dims=(0,1)`, sai —
    từng cho `dec(random z)` ra RMS ~7e-4 thay vì ~2.8e-2).

    Với Conv1d `[out, in, k]` lẫn ConvTranspose1d `[in, out, k]`, chuẩn hóa luôn
    bỏ chiều 0 nên một công thức dùng chung được.
    """
    dims = tuple(range(1, v.dim()))
    norm = torch.norm(v, p=2, dim=dims, keepdim=True)
    if tuple(g.shape) != tuple(norm.shape):
        raise ValueError(
            f"weight-norm: g {tuple(g.shape)} != norm {tuple(norm.shape)} — "
            "sai trục chuẩn hoá (phải là dims=(1,…,−1), KHÔNG phải 'trừ chiều cuối')"
        )
    return v * (g / norm)


# ---------------------------------------------------------------- lọc + gộp


def collect_dec(model_sd: dict) -> dict:
    """Trả về {tên_đã_bỏ_prefix: np.ndarray} cho mọi tensor của `dec`.

    Tự decompose weight-norm khi gặp cặp `weight_g` / `weight_v`:
    gộp thành `weight`, không xuất hai tensor gốc.
    """
    out: dict[str, np.ndarray] = {}
    skipped_v = 0
    for key, val in model_sd.items():
        if not key.startswith("dec."):
            continue
        name = key[len("dec.") :]

        if name.endswith("weight_g"):
            base = name[: -len("weight_g")]  # ví dụ "ups.0.", "resblocks.0.convs1.0."
            vk = "dec." + base + "weight_v"
            if vk in model_sd:
                out[base + "weight"] = wn_compose(
                    model_sd[key], model_sd[vk]
                ).numpy()
            continue

        if name.endswith("weight_v"):
            skipped_v += 1
            continue  # đã gộp ở nhánh weight_g

        out[name] = val.numpy()

    print(f"  bo qua {skipped_v} tensor weight_v (da gop vao weight)")
    return out


# ---------------------------------------------------------------- kiểm tra


def verify_structure(t: dict, hps: dict) -> list:
    """Trả về danh sách lỗi (rỗng = hợp lệ)."""
    err: list = []
    init = hps["upsample_initial_channel"]
    rates, kern = hps["upsample_rates"], hps["upsample_kernel_sizes"]
    rk, rd = hps["resblock_kernel_sizes"], hps["resblock_dilation_sizes"]

    # conv_pre
    if tuple(t["conv_pre.weight"].shape) != (init, 192, 7):
        err.append(f"conv_pre.weight {t['conv_pre.weight'].shape} != {(init, 192, 7)}")
    if tuple(t["conv_pre.bias"].shape) != (init,):
        err.append("conv_pre.bias sai shape")

    # conv_post — KHÔNG bias
    if "conv_post.bias" in t:
        err.append("conv_post.bias CÓ tồn tại — nhưng Conv1d bias=False!")
    ch_last = init // (2 ** len(rates))
    if tuple(t["conv_post.weight"].shape) != (1, ch_last, 7):
        err.append(
            f"conv_post.weight {t['conv_post.weight'].shape} != {(1, ch_last, 7)}"
        )

    # ups — ConvTranspose1d luu weight theo [in, out, k] (KHAC Conv1d la [out, in, k])
    for i, (u, k) in enumerate(zip(rates, kern)):
        exp = (init // (2**i), init // (2 ** (i + 1)), k)
        got = tuple(t[f"ups.{i}.weight"].shape)
        if got != exp:
            err.append(f"ups.{i}.weight {got} != {exp}")
        if tuple(t[f"ups.{i}.bias"].shape) != (exp[1],):  # bias theo out_channels
            err.append(f"ups.{i}.bias sai shape {tuple(t[f'ups.{i}.bias'].shape)}")

    # resblocks
    for i in range(len(rates)):
        ch = init // (2 ** (i + 1))
        for j, k in enumerate(rk):
            b = i * len(rk) + j
            for grp in ("convs1", "convs2"):
                for c in range(len(rd[j])):
                    name = f"resblocks.{b}.{grp}.{c}.weight"
                    if name not in t:
                        err.append(f"thiếu {name}")
                        continue
                    got = tuple(t[name].shape)
                    if got != (ch, ch, k):  # convs2 dùng CÙNG k
                        err.append(f"{name} {got} != {(ch, ch, k)}")

    return err


# ---------------------------------------------------------------- ground truth


def build_reference(t: dict, hps: dict):
    """Dựng lại module forward từ CHÍNH các tensor đã xuất.

    Cố ý không dùng model gốc: nếu lỗi ở khâu gộp weight-norm thì cách này
    vẫn cho kết quả đúng, và Rust sẽ khớp — ta sẽ không phát hiện được lỗi.
    Dùng model gốc thì ngược lại: xác minh được chuỗi gộp từ checkpoint tới
    reference đúng hay không.
    """
    import torch.nn as nn

    init = hps["upsample_initial_channel"]
    rates, kern = hps["upsample_rates"], hps["upsample_kernel_sizes"]
    rk, rd = hps["resblock_kernel_sizes"], hps["resblock_dilation_sizes"]
    LRELU = 0.1

    def pad(k, d):
        return (k * d - d) // 2

    class ResBlock(nn.Module):
        def __init__(self, ch, k, dil):
            super().__init__()
            self.convs1 = nn.ModuleList(
                [nn.Conv1d(ch, ch, k, 1, dilation=d, padding=pad(k, d)) for d in dil]
            )
            # convs2: CÙNG kernel_size, dilation=1  (bẫy số 2)
            self.convs2 = nn.ModuleList(
                [nn.Conv1d(ch, ch, k, 1, dilation=1, padding=pad(k, 1)) for _ in dil]
            )

        def forward(self, x):
            for c1, c2 in zip(self.convs1, self.convs2):
                r = x
                x = nn.functional.leaky_relu(x, LRELU)
                x = nn.functional.leaky_relu(c1(x), LRELU)
                x = c2(x) + r
            return x

    class Gen(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_pre = nn.Conv1d(192, init, 7, 1, padding=3)
            self.ups = nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        init // (2**i),
                        init // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                    for i, (u, k) in enumerate(zip(rates, kern))
                ]
            )
            self.resblocks = nn.ModuleList(
                [
                    ResBlock(init // (2 ** (i + 1)), rk[j], rd[j])
                    for i in range(len(rates))
                    for j in range(len(rk))
                ]
            )
            self.conv_post = nn.Conv1d(
                init // (2 ** len(rates)), 1, 7, 1, padding=3, bias=False
            )

        def forward(self, z):
            x = self.conv_pre(z)
            for i in range(len(rates)):
                x = nn.functional.leaky_relu(x, LRELU)
                x = self.ups[i](x)
                acc = None
                for j in range(len(rk)):
                    y = self.resblocks[i * len(rk) + j](x)
                    acc = y if acc is None else acc + y
                x = acc / len(rk)
            # LReLU CUỐI không truyền slope → PyTorch default 0.01 (bẫy số 1)
            x = nn.functional.leaky_relu(x)
            x = self.conv_post(x)
            return torch.tanh(x)

    net = Gen().eval()

    def put(param, name):
        param.data = torch.from_numpy(t[name].copy())

    put(net.conv_pre.weight, "conv_pre.weight")
    put(net.conv_pre.bias, "conv_pre.bias")
    for i in range(len(rates)):
        put(net.ups[i].weight, f"ups.{i}.weight")
        put(net.ups[i].bias, f"ups.{i}.bias")
    for b in range(len(net.resblocks)):
        for grp in ("convs1", "convs2"):
            for c in range(len(rd[0])):
                mod = getattr(net.resblocks[b], grp)[c]
                put(mod.weight, f"resblocks.{b}.{grp}.{c}.weight")
                put(mod.bias, f"resblocks.{b}.{grp}.{c}.bias")
    put(net.conv_post.weight, "conv_post.weight")

    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--hps", default=HPS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--ts", default="1,2,10,37,100,137,250",
                    help="do dai T de sinh ground truth (bien duoi, le, tron)")
    ap.add_argument("--no-ref", action="store_true",
                    help="chi xuat trong so, khong chay ground truth")
    args = ap.parse_args()

    with open(args.hps) as f:
        hps = json.load(f)["model"]
    ts = [int(x) for x in args.ts.split(",")]

    print(f"checkpoint: {args.ckpt}")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  top-level: {list(raw.keys())}")
    print(f"  tong tensor: {len(raw['model'])}")

    print("\n1) loc + gop weight-norm")
    tensors = collect_dec(raw["model"])
    print(f"  -> {len(tensors)} tensor")

    print("\n2) kiem tra cau truc")
    errs = verify_structure(tensors, hps)
    if errs:
        for e in errs:
            print("  x", e)
        raise SystemExit(1)
    print("  OK: conv_post khong bias · 4 ups · 12 resblock · convs2 cung kernel")

    print("\n3) ghi ra dia")
    d = os.path.join(args.out, "dec")
    os.makedirs(d, exist_ok=True)
    for name, arr in tensors.items():
        np.save(os.path.join(d, name + ".npy"), arr)
    nparam = int(sum(v.size for v in tensors.values()))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(
            {
                "initial_channel": 192,
                "hidden_channels": hps["hidden_channels"],
                "upsample_initial_channel": hps["upsample_initial_channel"],
                "upsample_rates": hps["upsample_rates"],
                "upsample_kernel_sizes": hps["upsample_kernel_sizes"],
                "resblock_kernel_sizes": hps["resblock_kernel_sizes"],
                "resblock_dilation_sizes": hps["resblock_dilation_sizes"],
                "lrelu_slope": 0.1,
                "lrelu_slope_final": 0.01,
                "conv_post_bias": False,
                "n_tensors": len(tensors),
                "n_params": nparam,
                "sampling_rate": 22050,
                "upsample_product": int(np.prod(hps["upsample_rates"])),
            },
            f,
            indent=2,
        )
    print(f"  {args.out}/dec/*.npy          {len(tensors)} file")
    print(f"  {args.out}/config.json")
    print(f"  tham so: {nparam / 1e6:.2f} M")

    if args.no_ref:
        return

    print("\n4) ground truth (forward tai day tu tensor da xuat)")
    net = build_reference(tensors, hps)
    torch.manual_seed(0)
    refs, zs = {}, {}
    for T in ts:
        z = torch.randn(1, 192, T)
        with torch.no_grad():
            y = net(z)
        exp = T * 256
        got = y.shape[-1]
        flag = "OK " if got == exp else "SAI"
        print(f"  T={T:<4d} -> {got:>7d} sample  (mong doi {exp:>7d})  {flag}")
        if got != exp:
            raise SystemExit(f"chieu dai sai o T={T}: {got} != {exp}")
        refs[f"T{T}"] = y.numpy()
        zs[f"T{T}"] = z.numpy()
    # Kiem tinh chat VAT LY: dec(z ngau nhien ~ N(0,1)) phai ra muc tieng noi.
    # Ban dau (truc weight-norm sai) `dec` bi co ~30x -> RMS ~7e-4, check nay bat.
    big = max(refs, key=lambda k: refs[k].shape[-1])
    rms = float(np.sqrt((refs[big] ** 2).mean()))
    print(f"  RMS ref {big} (ki vong ~2e-2): {rms:.3e}")
    if rms < 5e-3:
        raise SystemExit(
            f"dec ra qua yeu (RMS {rms:.3e}) — nghi truc weight-norm sai (xem §2.3.1)"
        )
    np.savez(os.path.join(args.out, "ref_audio.npz"), **refs)
    np.savez(os.path.join(args.out, "ref_input.npz"), **zs)
    print(f"  {args.out}/ref_audio.npz      ({len(ts)} nhom)")
    print(f"  {args.out}/ref_input.npz      z de tai")


if __name__ == "__main__":
    main()