"""Blend: the song untouched + the live band of ONE reference track, locked to the song's tempo and bars, fading in.

Usage (main env):
    python tools/blend_real.py <song-root> <reference> [--tag name] [--floor 0.35] [--level 1.0] [--shift auto|N]
<song-root> holds orig/{vocals,no_vocals,drums,...}.wav and native/song.json (run native-arrange once, or the UI);
<reference> is a folder name in <song-root>/reference/ (stems of the reference track, `check-tracks --demux`).
No slicing, no patterns: the reference's own backing (drums + bass + other) is time-warped so that its eighth notes
fall on the song's bar grid (a live band drifts, so the warp follows its tempo curve), pitch-shifted by whole
semitones only if that makes the common key clearer, and faded in over the song while the song's own backing fades
down to --floor. Vocals are never touched. Writes `<song-root>/native/out/<song>-blend-<reference>[-tag].wav`.
Judge by ear; the printed numbers only say how well the drums lock.
"""
import argparse
import os

import librosa
import numpy as np
import soundfile as sf

import native_arrange as na
import native_song as ns
from native_bank import SR, band_fractions, drum_class, load

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
HOP = 512
NFFT = 2048


def live_map(drums, bass):
    """Eighth-note clock of a live band: (S, c8) where S[K] is the time of its K-th eighth note and K = c8 (mod 8) is a bar start.

    Local tempo from 30 s windows (the drummer drifts), integrated; then the smoothed residual of the real hits
    around that clock is added, so the clock follows the hands, not a metronome."""
    dur = len(drums) / SR
    t, w = ns.hit_times(drums)
    cand = np.arange(100, 180, 0.02)
    centers, bpms = [], []
    for a in np.arange(0, max(dur - 30, 1) + 1, 15):
        m = (t >= a) & (t < a + 30)
        if m.sum() < 20:
            continue
        z = np.abs(np.exp(2j * np.pi * np.outer(cand, t[m]) * 2 / 60) @ w[m])
        centers.append(a + 15)
        bpms.append(cand[int(np.argmax(z))])
    tt = np.arange(0, dur, 0.005)
    p = np.cumsum(np.interp(tt, centers, bpms) / 30) * 0.005             # eighth notes since the start
    off = np.angle(np.sum(w * np.exp(2j * np.pi * np.interp(t, tt, p)))) / (2 * np.pi)
    q = p - off
    qi = np.interp(t, tt, q)
    K = np.round(qi).astype(int)
    ok = (np.abs(qi - K) < 0.3) & (K >= 0)
    kk = np.arange(0, int(q[-1]))
    delta = t - np.interp(K, q, tt)
    g = np.exp(-0.5 * ((kk[:, None] - K[None, ok.nonzero()[0]]) / 6.0) ** 2)
    wo = w[ok]
    den = g @ wo
    ds = np.where(den > 1e-3 * wo.sum(), (g @ (wo * delta[ok])) / np.maximum(den, 1e-12), 0.0)
    S = np.maximum.accumulate(np.interp(kk, q, tt) + ds)
    # bar start: the snare is on beats 2 and 4 (K = 2 mod 4 within a bar); kick-or-not can't tell bar from half bar,
    # so of the two candidates take the one where the bass lands harder
    sn = np.zeros(4)
    for ti, ki, wi, o in zip(t, K, w, ok):
        seg = drums[max(0, int((ti - 0.01) * SR)):int((ti + 0.06) * SR)]
        if o and len(seg) >= 512 and drum_class(band_fractions(seg)) == "snare":
            sn[ki % 4] += wi
    c4 = (int(np.argmax(sn)) - 2) % 4
    bt, bw = ns.hit_times(bass)
    bk = np.round(np.interp(bt, tt, q)).astype(int)
    sc = [bw[bk % 8 == c].sum() for c in (c4, c4 + 4)]
    c8 = c4 + 4 * int(sc[1] > sc[0])
    print(f"Live band: tempo {min(bpms):.1f}-{max(bpms):.1f} BPM, bar start at K == {c8} (mod 8), "
          f"snare confidence {sn.max() / max(sn.sum(), 1e-9):.2f}, bass {max(sc) / max(sum(sc), 1e-9):.2f}")
    return S, c8


def warp(y, sigma):
    """Phase vocoder with a time-varying rate: output frame m reads the source at time sigma[m] (seconds, increasing)."""
    out = []
    for ch in y:
        D = librosa.stft(ch, n_fft=NFFT, hop_length=HOP)
        D = np.pad(D, ((0, 0), (0, 2)))
        adv = 2 * np.pi * HOP * np.arange(D.shape[0]) / NFFT
        acc = np.angle(D[:, 0])
        res = np.zeros((D.shape[0], len(sigma)), complex)
        for m, s in enumerate(sigma * SR / HOP):
            i = int(s)
            if i >= D.shape[1] - 2:
                break
            a = s - i
            mag = (1 - a) * np.abs(D[:, i]) + a * np.abs(D[:, i + 1])
            res[:, m] = mag * np.exp(1j * acc)
            d = np.angle(D[:, i + 1]) - np.angle(D[:, i]) - adv
            acc = acc + adv + d - 2 * np.pi * np.round(d / (2 * np.pi))
        out.append(librosa.istft(res, hop_length=HOP, length=len(sigma) * HOP))
    return np.array(out)


def clarity(chroma):
    return max(np.corrcoef(chroma, np.roll(pr, i))[0, 1] for pr in (MAJOR, MINOR) for i in range(12))


def chroma(y):
    c = librosa.feature.chroma_cqt(y=librosa.resample(y.mean(0), orig_sr=SR, target_sr=22050), sr=22050).mean(1)
    return c / c.sum()


def best_shift(song, band):
    """Whole semitones (-3..3) of the band that make the combined key clearest (a weak hint, `--shift N` overrides)."""
    a, b = chroma(song), chroma(band)
    res = {k: clarity(a + np.roll(b, k)) for k in range(-3, 4)}
    print("Clarity of the common key by the band's shift (semitones):",
          {k: round(float(v), 3) for k, v in res.items()}, "| the song alone:", round(float(clarity(a)), 3))
    return max(res, key=lambda k: (round(res[k], 3), -abs(k)))


def lock_report(ref_drums, song_drums):
    """Onset-envelope correlation of the song's drums and the warped reference drums, by thirds of the song."""
    h = 256
    e1 = librosa.onset.onset_strength(y=song_drums, sr=SR, hop_length=h)
    e2 = librosa.onset.onset_strength(y=ref_drums, sr=SR, hop_length=h)
    n = min(len(e1), len(e2))
    e1, e2 = e1[:n], e2[:n]
    lags = range(-int(0.06 * SR / h), int(0.06 * SR / h) + 1)
    for part in range(3):
        a, z = part * n // 3, (part + 1) * n // 3
        x = e1[a:z] - e1[a:z].mean()
        cs = {l: np.corrcoef(x, np.roll(e2[a:z], l) - np.roll(e2[a:z], l).mean())[0, 1] for l in lags}
        best = max(cs, key=cs.get)
        print(f"  part {part + 1}: hit correlation {cs[0]:.2f} at zero, best {cs[best]:.2f} at a shift of {best * h / SR * 1000:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("reference")
    ap.add_argument("--tag", default="")
    ap.add_argument("--floor", type=float, default=0.35, help="level the song's own backing fades down to")
    ap.add_argument("--level", type=float, default=1.0, help="level the reference reaches, 1 = as loud as the song's backing")
    ap.add_argument("--shift", default="auto")
    ap.add_argument("--fade", default="0.05,0.9", help="where the fade-in starts and ends, as fractions of the song")
    a = ap.parse_args()
    ref = os.path.join(a.root, "reference", a.reference)
    song = {n: load(os.path.join(a.root, "orig", f"{n}.wav")) for n in ("vocals", "no_vocals", "drums")}
    band = {n: load(os.path.join(ref, f"{n}.wav")).T for n in ("no_vocals", "drums")}
    grid = na.analyse(a.root)["grid"]
    n_out = len(song["vocals"])
    L = n_out / SR

    S, c8 = live_map(band["drums"].mean(0), load(os.path.join(ref, "bass.wav")).mean(1))
    step = grid["bar"] / 8
    k0 = int(np.ceil(grid["bar0"] / step))
    k0 += (c8 - k0) % 8                                    # the reference's first used eighth is a bar start
    tm = np.arange(0, n_out // HOP + 1) * HOP / SR
    kf = k0 + (tm - grid["bar0"]) / step
    sigma = np.interp(kf, np.arange(len(S)), S)
    if kf[-1] > len(S) - 1:
        print(f"Warning: the band is shorter than the song, silence after {S[-1]:.0f} s.")
    print("Fitting the band to the song grid...")
    mix_band = warp(band["no_vocals"], sigma)[:, :n_out]
    drums_band = warp(band["drums"], sigma)[:, :n_out]
    print("Match of the band drums with the song drums:")
    lock_report(drums_band.mean(0), song["drums"].mean(1))

    k = best_shift(song["vocals"].T + song["no_vocals"].T, mix_band) if a.shift == "auto" else int(a.shift)
    if k:
        mix_band = librosa.effects.pitch_shift(mix_band, sr=SR, n_steps=k)
    print(f"Band shift: {k:+d} semitones.")

    t = np.arange(n_out) / SR
    f0, f1 = (float(v) for v in a.fade.split(","))
    x = np.clip((t - f0 * L) / ((f1 - f0) * L), 0, 1)
    s = x * x * (3 - 2 * x)                                # smoothstep: imagined -> real, no jumps
    song_bk = song["no_vocals"].T
    rms = lambda y: float(np.sqrt(np.mean(y ** 2)))
    g_band = a.level * rms(song_bk) / max(rms(mix_band), 1e-9)
    out = (song["vocals"].T + song_bk * (1 - (1 - a.floor) * s) + mix_band * g_band * s).T
    peak = float(np.abs(out).max())
    if peak > 0.97:
        out *= 0.97 / peak
    name = f"{os.path.basename(os.path.normpath(a.root))}-blend-{a.reference}" + (f"-{a.tag}" if a.tag else "") + ".wav"
    path = os.path.join(a.root, "native", "out", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, out, SR, subtype="PCM_16")
    print(f"Peak before limiting {peak:.2f}; band level x{g_band:.2f}.")
    print(path)


if __name__ == "__main__":
    main()
