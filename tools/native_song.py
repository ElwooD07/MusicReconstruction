"""Analysis of the song itself for route B: tempo grid, drum pattern, chord roots, energy.

Used by native_arrange.py (and analyze_structure.py for the grid). Everything is derived from the song's own stems
(`<root>/orig/{drums,bass,other,vocals}.wav`), nothing is generated.
"""
import librosa
import numpy as np
import scipy.signal as sg

from native_bank import HOP, NOTES, SR, band_fractions, drum_class, load

STEPS = 16                                       # grid steps per bar (sixteenth notes in 4/4)


def hit_times(mono, hop=128):
    """Onset peaks (not back-tracked: the grid needs the moment of the hit, not of the quiet before it) and strengths."""
    env = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=hop)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=hop, backtrack=False)
    return librosa.frames_to_time(fr, sr=SR, hop_length=hop), env[fr]


def fit_grid(drums, bass=None, other=None, lo=90.0, hi=200.0):
    """Sixteenth-note grid of a metronomic song: dict(bpm, step, t0, bar, coherence).

    The tempo is where the drum onsets line up best with a sixteenth grid (phase coherence, scanned finely), refined by
    a regression of onset times on grid indices. A beat tracker is NOT used: on this kind of music it is off by a few
    percent, which is enough to drift by whole beats within a song. t0 is a grid point; the bar start `bar0` (also a
    grid point) comes from the drum pattern (kick on 1, snare on 2 and 4), the bass onsets and where the chords change.
    """
    t, w = hit_times(drums)
    bpms = np.arange(lo, hi, 0.01)
    z = np.array([abs(np.sum(w * np.exp(2j * np.pi * t * 4 * b / 60)) / w.sum()) for b in bpms])
    bpm = float(bpms[int(np.argmax(z))])
    step = 60 / bpm / 4
    t0 = (np.angle(np.sum(w * np.exp(-2j * np.pi * t / step))) / (2 * np.pi) * step) % step
    for _ in range(3):
        k = np.round((t - t0) / step)
        r = (t - t0) - k * step
        ok = np.abs(r) < 0.25 * step
        step, t0 = np.polyfit(k[ok], t[ok], 1)
    k = np.round((t - t0) / step)
    coherence = float(np.mean(np.abs((t - t0) - k * step) < 0.15 * step))
    grid = dict(bpm=60 / step / 4, step=float(step), t0=float(t0), bar=float(step * STEPS), coherence=coherence)
    o = bar_offset(drums, bass, grid)
    if other is not None:
        o = harmonic_parity(other, grid, o)
    grid["bar0"] = (grid["t0"] + o * grid["step"]) % grid["bar"]      # the first bar line in the song
    return grid


def bar_offset(drums, bass, grid):
    """Which of the 16 grid steps is the first step of a bar: kick on 1, snare on 2 and 4, bass changes on 1."""
    t, _ = hit_times(drums)
    k = np.round((t - grid["t0"]) / grid["step"]).astype(int)
    score = np.zeros(STEPS)
    for ti, ki in zip(t, k):
        seg = drums[max(0, int((ti - 0.01) * SR)):int((ti + 0.06) * SR)]
        if len(seg) < 512:
            continue
        c = drum_class(band_fractions(seg))
        for o in range(STEPS):
            pos = (ki - o) % STEPS
            if c == "kick" and pos == 0:
                score[o] += 1.0
            if c == "snare" and pos in (4, 12):
                score[o] += 1.0
    if bass is not None:
        bt, bw = hit_times(bass)
        bk = np.round((bt - grid["t0"]) / grid["step"]).astype(int)
        for ki, wi in zip(bk, bw):
            for o in range(STEPS):
                if (ki - o) % STEPS == 0:
                    score[o] += 0.5 * wi / bw.mean()
    return int(np.argmax(score))


def harmonic_parity(other, grid, o):
    """The drums cannot tell a bar line from the middle of the bar (kick on 1 and 3, snare on 2 and 4): pick o or o + 8,
    whichever has the chord changes of `other` on it."""
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=other, sr=SR, hop_length=hop)
    half = 8 * grid["step"]

    def mean(a, b):
        return chroma[:, max(0, int(a * SR / hop)):max(1, int(b * SR / hop))].mean(1)

    change = []
    for p in (o % 8, o % 8 + 8):
        t, d = grid["t0"] + p * grid["step"], []
        while t + half < len(other) / SR:
            if t - half > 0:
                x, y = mean(t - half, t), mean(t, t + half)
                d.append(1 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)))
            t += 2 * half
        change.append(np.mean(d))
    return o % 8 + (8 if change[1] > change[0] else 0)


def step_of(t, grid):
    return (t - grid["bar0"]) / grid["step"]


def transcribe_drums(drums, grid):
    """Drum hits of the song: list of (step index from bar0, class, velocity 0..1, offset in steps from the grid)."""
    t, w = hit_times(drums)
    peak = []
    out = []
    for ti in t:
        seg = drums[max(0, int((ti - 0.01) * SR)):int((ti + 0.06) * SR)]
        if len(seg) < 512:
            continue
        peak.append(float(np.abs(seg).max()))
        out.append((ti, drum_class(band_fractions(seg)), peak[-1]))
    top = np.percentile(peak, 95)
    hits = []
    for ti, c, p in out:
        if c not in ("kick", "snare"):
            continue
        s = step_of(ti, grid)
        k = int(round(s))
        hits.append((k, c, float(min(1.0, p / top)), float(s - k)))
    # hats: onsets of the band above 7 kHz (they ride on top of kicks and snares, so a spectrum class would miss them)
    hp = sg.sosfiltfilt(sg.butter(4, 7000, "hp", fs=SR, output="sos"), drums)
    ht, hw = hit_times(hp)
    floor, top = 0.25 * np.median(hw), np.percentile(hw, 95)
    for ti, wi in zip(ht, hw):
        if wi < floor:
            continue
        s = step_of(ti, grid)
        k = int(round(s))
        hits.append((k, "hat", float(min(1.0, wi / top)), float(s - k)))
    hits.sort()
    return hits


def chord_roots(bass, other, grid, n_bars, per_bar=2):
    """Root pitch class for every half bar. Evidence: the guitar-like root + fifth in `other` and the bass note, each
    whitened by the song's overall chroma. A bar gets one chord unless a half bar clearly disagrees with it."""
    hop = 512
    cb = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=hop, fmin=librosa.note_to_hz("C1"))
    co = librosa.feature.chroma_cqt(y=other, sr=SR, hop_length=hop)
    cb = cb / cb.mean(1, keepdims=True)                        # whiten: notes the whole song leans on must not win
    co = co / co.mean(1, keepdims=True)
    seg = grid["bar"] / per_bar
    obs = []
    for i in range(n_bars * per_bar):
        a, b = int((grid["bar0"] + i * seg) * SR / hop), int((grid["bar0"] + (i + 1) * seg) * SR / hop)
        if b <= a or a >= cb.shape[1]:
            v = np.ones(12)
        else:
            x = cb[:, a:b].mean(1)
            y = co[:, a:b].mean(1)
            x, y = x / x.sum(), y / y.sum()
            v = x + 0.6 * np.array([y[r] + 0.8 * y[(r + 7) % 12] for r in range(12)]) / 1.8
        obs.append(v / v.sum())
    roots, conf = [], []
    for i in range(0, len(obs), per_bar):
        halves = obs[i:i + per_bar]
        bar_root = int(np.argmax(np.sum(halves, axis=0)))
        for v in halves:
            r = int(np.argmax(v))
            roots.append(r if v[r] > 1.4 * v[bar_root] else bar_root)
            conf.append(float(v[roots[-1]]))
    return roots, conf


def bar_energy(other, vocals, drums, grid, n_bars):
    """Per bar: RMS of the song's own backing layer, of the vocal and of the drums (linear)."""
    out = []
    for b in range(n_bars):
        a, z = int((grid["bar0"] + b * grid["bar"]) * SR), int((grid["bar0"] + (b + 1) * grid["bar"]) * SR)
        rms = lambda y: float(np.sqrt(np.mean(y[a:z] ** 2))) if z > a else 0.0
        out.append(dict(other=rms(other), vocals=rms(vocals), drums=rms(drums)))
    return out
