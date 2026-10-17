#!/usr/bin/env python3
"""Dump reference log-mel for the SpeechT5 HiFi-GAN vocoder frontend.

Produces `n_frames = 1 + len(wav)/hop` frames, matching HF
`SpeechT5FeatureExtractor` exactly:
  n_fft=1024, hop=256, win=1024, hann(periodic), center=True reflect pad,
  80 mel bins, fmin=80, fmax=7600, Slaney scale + Slaney norm,
  amplitude (power=1.0), floor 1e-10, log10.
"""

import numpy as np

SR = 16000
N_MELS = 80
N_FFT = 1024
HOP = 256
WIN = 1024
FMIN = 80.0
FMAX = 7600.0
MEL_FLOOR = 1e-10

OUT = "sources/ref_mel.npy"
WAV = "sources/422-122949-0031.wav"


def read_float_wav(path):
    """Parse a minimal IEEE-float (format 3) WAV, returning mono float32."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 12
    fmt = None
    pcm = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        (size,) = np.frombuffer(data[pos + 4 : pos + 8], dtype="<u4").tolist()
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = np.frombuffer(body, dtype="<u2")
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise SystemExit("missing fmt/data chunk")
    channels = int(fmt[1])
    rate = int(fmt[2])
    bits = int(fmt[7])
    if fmt[0] != 3 or bits != 32:
        raise SystemExit("expected float32 WAV, got format=%d bits=%d" % (fmt[0], bits))
    samples = np.frombuffer(pcm, dtype="<f4").astype(np.float64)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def hann_periodic(n):
    # torch.hann_window(periodic=True) == np.hanning(n + 1)[:-1]
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def dominant_fft_length(frame_length):
    """HF `optimal_fft_length`: smallest power of two >= frame_length."""
    n = 1
    while n < frame_length:
        n *= 2
    return n


def hz_to_mel_slaney(freq):
    # librosa Slaney mel (htk=False), identical to HF audio_utils
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    if isinstance(mels, np.ndarray):
        log_t = mels >= min_log_mel
        mels[log_t] = min_log_mel + np.log(freq[log_t] / min_log_hz) / logstep
    elif mels >= min_log_mel:
        mels = min_log_mel + np.log(freq / min_log_hz) / logstep
    return mels


def mel_to_hz_slaney(mels):
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    if isinstance(mels, np.ndarray):
        # NOTE: librosa/HF test the *mel* value, not the linear freqs.
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def mel_filter_bank(n_mels, n_freqs, sr, fmin, fmax):
    """HF audio_utils.mel_filter_bank with mel_scale='slaney', norm='slaney'."""
    fft_freqs = np.linspace(0, sr // 2, n_freqs)
    mel_min = hz_to_mel_slaney(fmin)
    mel_max = hz_to_mel_slaney(fmax)
    mel_freqs = np.linspace(mel_min, mel_max, n_mels + 2)
    filter_freqs = mel_to_hz_slaney(mel_freqs)

    # _create_triangular_filter_bank (adapted from torchaudio/librosa)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(0.0, np.minimum(down_slopes, up_slopes))  # [n_freqs, n_mels]

    # Slaney norm
    enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
    filters *= np.expand_dims(enorm, 0)
    return filters  # [n_freqs, n_mels]


def main():
    wav, sr = read_float_wav(WAV)
    if sr != SR:
        raise SystemExit("expected %d Hz, got %d" % (SR, sr))
    print("wav: %d samples, %.3f s" % (wav.size, wav.size / SR))

    n_fft = dominant_fft_length(WIN)
    window = hann_periodic(WIN)
    n_freqs = n_fft // 2 + 1

    # center=True, pad_mode='reflect' -> pad n_fft/2 each side
    pad = n_fft // 2
    padded = np.pad(wav, pad, mode="reflect")
    n_frames = 1 + np.floor((padded.size - WIN) / HOP).astype(int)
    print("n_fft=%d n_freqs=%d n_frames=%d" % (n_fft, n_freqs, n_frames))

    spec = np.empty((n_frames, n_freqs), dtype=np.complex64)
    for t in range(n_frames):
        s = t * HOP
        frame = padded[s : s + WIN] * window
        spec[t] = np.fft.rfft(frame, n=n_fft)

    magnitude = np.abs(spec)  # power = 1.0
    mel_filters = mel_filter_bank(N_MELS, n_freqs, SR, FMIN, FMAX)
    mel = np.maximum(MEL_FLOOR, np.dot(magnitude, mel_filters))
    log_mel = np.log10(mel).astype(np.float32)

    np.save(OUT, log_mel.T)  # [n_mels, n_frames]
    print("wrote %s shape=%s  min=%.4f max=%.4f"
          % (OUT, log_mel.T.shape, log_mel.min(), log_mel.max()))


if __name__ == "__main__":
    main()
