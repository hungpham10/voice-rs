#!/usr/bin/env python3
"""Cross-check scripts/ref_mel.py against the real HF feature extractor.

If this passes, `sources/ref_mel.npy` is a trustworthy ground truth to compare
the Rust `voice-mel` implementation against.
"""

import sys

import numpy as np

sys.path.insert(0, "scripts")
import ref_mel as R  # noqa: E402


def main() -> int:
    import torch
    from transformers import SpeechT5FeatureExtractor

    wav, sr = R.read_float_wav(R.WAV)
    wav32 = wav.astype(np.float32)

    feat = SpeechT5FeatureExtractor()
    out = feat([wav32], sampling_rate=16000, return_tensors="pt")
    hf_mel = out.input_features[0].numpy()  # [n_mels, n_frames]
    print("HF  : sample_size=%d stride=%d n_fft=%d shape=%s"
          % (feat.sample_size, feat.sample_stride, feat.n_fft, hf_mel.shape))
    print("HF  : fmin=%s fmax=%s floor=%s norm=%s scale=%s"
          % (feat.fmin, feat.fmax, feat.mel_floor, "?", "?"))

    ref = np.load("sources/ref_mel.npy")
    print("ref : shape=%s" % (ref.shape,))

    if ref.shape != hf_mel.shape:
        print("FAIL: shape mismatch")
        return 1
    diff = np.abs(ref.astype(np.float64) - hf_mel.astype(np.float64))
    denom = np.maximum(np.abs(hf_mel), 1e-8)
    print("max abs diff = %.3e   max rel diff = %.3e" % (diff.max(), (diff / denom).max()))
    ok = diff.max() < 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
