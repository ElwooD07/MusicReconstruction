"""Objective comparison of generated backing tracks against the source stem.

Usage (main env):
    python tools/compare_outputs.py --vocals <vocals.wav> <source.wav> <label=path.wav> ...
All numbers are on the same excerpt. Columns:
  onset_corr / lag_ms : correlation of onset-strength envelopes with the source (best lag within
                        +-80 ms) - do the hits fall where the source's hits fall? lag should be ~0
  chroma              : harmony similarity to the source (1 = identical)
  gtr_dip%            : % of 0.5 s windows where the guitar band (400-4000 Hz) drops >12 dB under its
                        median - "the guitar stops". Lower is better, the source's value is the reference
  perc_gap_dB         : percussive loudness in vocal gaps minus under the voice. Positive = drums are
                        heavier only between vocal phrases (wanted)
  gtr_voc_dB          : guitar-band loudness while the voice sings minus in gaps. Positive = fuller
                        guitar layers around the voice (wanted)
  nov_p90             : timbre jumps (MFCC 2 s before vs after), lower = smoother
"""
import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 22050
HOP = 512
WIN = SR // 2  # 0.5 s windows for the vocal-aware metrics


def load(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR)


def novelty(y, w=8):
    hop = SR // 4
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, hop_length=hop, n_fft=2048)[1:]
    return np.array([np.linalg.norm(mf[:, i - w:i].mean(1) - mf[:, i:i + w].mean(1))
                     for i in range(w, mf.shape[1] - w)])


def chroma_cos(y, src):
    a = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=2048)
    b = librosa.feature.chroma_stft(y=src, sr=SR, hop_length=2048)
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    return float(((a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)).mean())


def onset_corr(y, src, max_lag_ms=80):
    a = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    b = librosa.onset.onset_strength(y=src, sr=SR, hop_length=HOP)
    n = min(len(a), len(b))
    a, b = (a[:n] - a[:n].mean()) / (a[:n].std() + 1e-9), (b[:n] - b[:n].mean()) / (b[:n].std() + 1e-9)
    frame_ms = 1000 * HOP / SR
    best = (-1.0, 0)
    for lag in range(-int(max_lag_ms / frame_ms), int(max_lag_ms / frame_ms) + 1):
        x, z = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        c = float((x * z).mean())
        if c > best[0]:
            best = (c, round(lag * frame_ms))
    return best


def pitch_shift(y, src):
    """Semitone shift that best aligns the average chroma of y with the source (0 = same key area),
    and the correlation at that shift (how clearly the pitch content agrees)."""
    a = librosa.feature.chroma_cqt(y=y, sr=SR).mean(1)
    b = librosa.feature.chroma_cqt(y=src, sr=SR).mean(1)
    corr = [np.corrcoef(np.roll(a, -k), b)[0, 1] for k in range(12)]
    k = int(np.argmax(corr))
    return (k if k <= 6 else k - 12), float(corr[k])


def spectrum_cos(y, src):
    """Cosine similarity of the long-term spectrum in 1/3-octave-ish log bands (frequency balance)."""
    def bands(x):
        S = np.abs(librosa.stft(x, n_fft=4096)).mean(1)
        f = librosa.fft_frequencies(sr=SR, n_fft=4096)
        edges = 40 * 2 ** (np.arange(0, 28) / 3)
        return np.array([S[(f >= lo) & (f < hi)].mean() if ((f >= lo) & (f < hi)).any() else 0
                         for lo, hi in zip(edges[:-1], edges[1:])])
    a, b = np.log1p(bands(y) * 100), np.log1p(bands(src) * 100)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def restyle(y, src):
    """(timbre distance, bite dB): how far the overall timbre moved from the source (MFCC means) and how
    much more/less energy sits in 2-8 kHz relative to the total (positive = brighter/harsher/'heavier')."""
    def feats(x):
        mf = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=20)[1:].mean(1)
        S = np.abs(librosa.stft(x, n_fft=2048)) ** 2
        f = librosa.fft_frequencies(sr=SR, n_fft=2048)
        return mf, 10 * np.log10(S[(f >= 2000) & (f < 8000)].sum() / S.sum())
    (ma, ha), (mb, hb) = feats(y), feats(src)
    return float(np.linalg.norm(ma - mb)), float(ha - hb)


def level_db(y, src):
    return float(20 * np.log10((np.sqrt((y ** 2).mean()) + 1e-9) / (np.sqrt((src ** 2).mean()) + 1e-9)))


def rms_db(y):
    return 20 * np.log10(librosa.feature.rms(y=y, hop_length=WIN)[0] + 1e-6)


def vocal_activity(vocals):
    env = librosa.feature.rms(y=vocals, hop_length=WIN)[0]
    return env > 0.15 * np.percentile(env, 90)


def vocal_aware(y, active):
    """(guitar-band dip %, percussive gap-minus-voice dB, guitar-band voice-minus-gap dB)."""
    mid = sosfilt(butter(4, [400, 4000], btype="band", fs=SR, output="sos"), y)
    mid_db = rms_db(mid)
    perc_db = rms_db(librosa.effects.percussive(y))
    n = min(len(mid_db), len(active))
    mid_db, perc_db, act = mid_db[:n], perc_db[:n], active[:n]
    dip = 100 * np.mean(mid_db < np.median(mid_db) - 12)
    if act.all() or not act.any():
        return dip, float("nan"), float("nan")
    return (dip, perc_db[~act].mean() - perc_db[act].mean(), mid_db[act].mean() - mid_db[~act].mean())


def main():
    args = sys.argv[1:]
    assert args[0] == "--vocals", "pass --vocals <vocals.wav> first"
    active = vocal_activity(load(args[1]))
    src = load(args[2])
    print(f"vocals active in {100 * active.mean():.0f}% of windows")
    print(f"{'label':<30}{'onset_corr':>11}{'lag_ms':>7}{'chroma':>8}{'shift_st':>9}{'pitch_r':>8}"
          f"{'timbre_d':>9}{'lvl_dB':>7}{'gtr_dip%':>9}{'perc_gap':>9}{'bite_dB':>8}{'nov_p90':>8}")
    for item in args[3:]:
        label, path = item.split("=", 1)
        y = load(path)
        c, lag = onset_corr(y, src)
        dip, perc_gap, gtr_voc = vocal_aware(y, active)
        shift, pr = pitch_shift(y, src)
        td, bite = restyle(y, src)
        print(f"{label:<30}{c:>11.2f}{lag:>7}{chroma_cos(y, src):>8.2f}{shift:>9}{pr:>8.2f}"
              f"{td:>9.1f}{level_db(y, src):>7.1f}{dip:>9.1f}"
              f"{perc_gap:>9.1f}{bite:>8.1f}{np.percentile(novelty(y), 90):>8.1f}")


if __name__ == "__main__":
    main()
