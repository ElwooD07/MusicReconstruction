"""Slice the stems of reference tracks into a bank of REAL samples (route B: no generation).

Usage (main env):
    python tools/native_bank.py <reference-dir> [<reference-dir> ...] [--bank <dir>] [--redo]
A <reference-dir> is `<root>/reference/<name>/` (made by demux / check-tracks --demux: drums.wav, other.wav, bass.wav)
or the whole `<root>/reference` folder (every track in it).
Writes `<bank>/<name>/drums/{kick,snare,hat}_NN.wav`, `<bank>/<name>/guitar/<root>_<len>_NN.wav`, `<bank>/<name>/bass/<note>_<len>_NN.wav` and
`<bank>/<name>/index.json` (default <bank> = `<root>/native/bank`). Every sample is a plain wav: listen to the bank and
delete what sounds bad, the arranger uses whatever is left. A finished track folder is skipped unless --redo.

Drums: onsets of drums.wav, each hit classified by its spectrum (kick low, snare body/noise, hat high), only clean hits
(one instrument, not cut short by the next hit). Guitar: onsets of other.wav, each slice labelled by the chord in it
(power chord: root = pitch class where root + fifth dominate the chroma, confirmed by the bass stem; thirds are too
weak in distorted guitar to tell minor from major, so the bank holds power chords only).
"""
import argparse
import json
import os
import shutil

import librosa
import numpy as np
import soundfile as sf

SR = 44100
HOP = 256
PRE = 0.004                      # slice starts this much before the onset
CUT = 0.004                      # slice ends this much before the next onset
FADE = 0.003
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LEN_CLASSES = (("short", 0.0, 0.22), ("mid", 0.22, 0.5), ("long", 0.5, 9.0))
KEEP_DRUMS = {"kick": 8, "snare": 8, "hat": 10, "crash": 3}
KEEP_PER_BUCKET = 3
MAX_GUITAR_S = 1.6
MAX_BASS_S = 1.2
MIN_DOMINANCE = 0.35


def load(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        y = librosa.resample(y.T, orig_sr=sr, target_sr=SR).T
    return y


def onsets(mono):
    env = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=HOP)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=False)
    fr = librosa.onset.onset_backtrack(fr, env)
    return librosa.frames_to_time(np.unique(fr), sr=SR, hop_length=HOP)


def band_fractions(seg):
    """Energy shares (low, body, mid, high) of a short window: tells kick / snare / hat apart."""
    f = np.fft.rfftfreq(8192, 1 / SR)
    x = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=8192)) ** 2

    def e(lo, hi):
        return x[(f >= lo) & (f < hi)].sum()
    tot = e(30, 16000) + 1e-12
    return np.array([e(30, 170), e(170, 600), e(600, 5000), e(5000, 16000)]) / tot


def drum_class(frac):
    low, body, mid, high = frac
    if low > 0.55 and high < 0.1:
        return "kick"
    if high > 0.5 and low < 0.05:
        return "hat"
    if low < 0.3 and body + mid > 0.5 and high < 0.5:
        return "snare"
    return None


def classify_drum_onsets(mono, times):
    """[(time, class or None, peak)] for the onsets `times` of a drum stem."""
    out = []
    for t in times:
        seg = mono[int(t * SR):int((t + 0.07) * SR)]
        if len(seg) < 512:
            continue
        out.append((float(t), drum_class(band_fractions(seg)), float(np.abs(seg).max())))
    return out


def cut(y, a, b):
    """y[a:b] (seconds) with a short fade-in and fade-out."""
    s = y[max(0, int(a * SR)):int(b * SR)].copy()
    n = min(int(FADE * SR), len(s) // 2)
    if n:
        s[:n] *= np.linspace(0, 1, n)[:, None]
        s[-n:] *= np.linspace(1, 0, n)[:, None]
    return s


def normalise(s, peak=0.9):
    return s * (peak / max(float(np.abs(s).max()), 1e-6))


def build_drums(ref, out):
    y = load(os.path.join(ref, "drums.wav"))
    mono = y.mean(1)
    times = onsets(mono)
    hits = classify_drum_onsets(mono, times)
    nxt = {t: (times[i + 1] if i + 1 < len(times) else len(mono) / SR) for i, t in enumerate(times)}
    index = []
    for cls, keep in KEEP_DRUMS.items():
        cand = []
        for t, c, peak in hits:
            if c != ("hat" if cls == "crash" else cls):
                continue
            room = nxt[t] - t - CUT
            if cls == "crash":
                if room < 0.9 or peak < 0.05:       # a cymbal that is allowed to ring out
                    continue
                max_len, need = 2.0, 0.9
            elif cls == "hat":
                max_len, need = 0.25, 0.03
                if room > 0.9:
                    continue
            else:
                max_len, need = 0.5, 0.18
            if room < need or peak < 0.05:
                continue
            seg = mono[int(t * SR):int((t + 0.07) * SR)]
            cand.append((t, min(room, max_len), peak, band_fractions(seg)))
        if not cand:
            continue
        med = np.median([c[3] for c in cand], axis=0)
        # clean = close to the typical spectrum of its class, loud, and with a long enough tail
        score = [-np.abs(c[3] - med).sum() + 0.5 * np.log10(c[2]) + 0.4 * min(c[1], 0.3) for c in cand]
        for k, i in enumerate(np.argsort(score)[::-1][:keep]):
            t, length, peak, _ = cand[i]
            name = f"{cls}_{k:02d}.wav"
            s = normalise(cut(y, t - PRE, t + length))
            sf.write(os.path.join(out, "drums", name), s, SR)
            index.append(dict(file=f"drums/{name}", kind=cls, dur=len(s) / SR, peak=peak))
    return index


def build_guitar(ref, out):
    y = load(os.path.join(ref, "other.wav"))
    mono = y.mean(1)
    bass = load(os.path.join(ref, "bass.wav")).mean(1)
    times = onsets(mono)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=mono, sr=SR, hop_length=hop)
    bchroma = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=hop)
    cands = []
    for i, t in enumerate(times):
        end = times[i + 1] - CUT if i + 1 < len(times) else len(mono) / SR
        length = min(end - t, MAX_GUITAR_S)
        if length < 0.1:
            continue
        a, b = int((t + 0.03) * SR / hop), int((t + min(0.4, length)) * SR / hop)
        if b <= a:
            continue
        c = chroma[:, a:b].mean(1)
        c = c / max(c.sum(), 1e-9)
        # power chord = root + fifth: the root is the pitch class where both are strong, and the bass must agree
        root = int(np.argmax([c[r] + 0.8 * c[(r + 7) % 12] for r in range(12)]))
        dominance = float(c[root] + c[(root + 7) % 12])     # share of the two notes in all 12 (uniform = 0.17)
        bc = bchroma[:, a:b].mean(1)
        if dominance < MIN_DOMINANCE or (bc.max() > 0.2 * bchroma.max() and int(np.argmax(bc)) != root):
            continue
        seg = mono[int(t * SR):int((t + min(0.3, length)) * SR)]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 0.01:
            continue
        cands.append(dict(t=float(t), length=float(length), root=root, score=dominance, rms=rms))
    buckets = {}
    for c in cands:
        lc = next(n for n, lo, hi in LEN_CLASSES if lo <= c["length"] < hi)
        buckets.setdefault((c["root"], lc), []).append(c)
    index = []
    for (root, lc), items in sorted(buckets.items()):
        items.sort(key=lambda c: -(c["score"] + 0.3 * np.log10(c["rms"])))
        for k, c in enumerate(items[:KEEP_PER_BUCKET]):
            name = f"{NOTES[root]}_{lc}_{k:02d}.wav"
            s = cut(y, c["t"] - PRE, c["t"] + c["length"])
            s = s * (0.1 / max(float(np.sqrt(np.mean(s[:int(0.3 * SR)] ** 2))), 1e-6))   # same loudness for all
            s = s * min(1.0, 0.95 / max(float(np.abs(s).max()), 1e-6))
            sf.write(os.path.join(out, "guitar", name), s, SR)
            index.append(dict(file=f"guitar/{name}", kind="guitar", root=root, len_class=lc, dur=len(s) / SR))
    return index, len(cands)


def bass_notes(mono):
    """Notes of a monophonic bass stem: [dict(t, length, midi, rms, voiced)] (pitch = median of the voiced frames)."""
    sr = 22050
    y = librosa.resample(mono, orig_sr=SR, target_sr=sr)
    hop = 512
    f0, _, _ = librosa.pyin(y, fmin=30, fmax=400, sr=sr, frame_length=2048, hop_length=hop)
    times = onsets(mono)
    notes = []
    for i, t in enumerate(times):
        end = times[i + 1] if i + 1 < len(times) else len(mono) / SR
        length = end - t
        a, b = int((t + 0.04) * sr / hop), int((t + min(0.35, max(length, 0.08))) * sr / hop)
        seg = f0[a:max(b, a + 1)]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 3:
            continue
        w = mono[int(t * SR):int((t + min(0.3, length)) * SR)]
        notes.append(dict(t=float(t), length=float(length), midi=int(round(float(librosa.hz_to_midi(np.median(seg))))),
                          rms=float(np.sqrt(np.mean(w ** 2))), voiced=len(seg) / max(b - a, 1)))
    return notes


def build_bass(ref, out):
    """Clean single bass notes of a reference (pitch by pyin), the best two per (pitch, length class)."""
    y = load(os.path.join(ref, "bass.wav"))
    mono = y.mean(1)
    buckets = {}
    for n in bass_notes(mono):
        length = min(n["length"] - CUT, MAX_BASS_S)
        if length < 0.1 or n["voiced"] < 0.6 or n["rms"] < 0.01:
            continue
        lc = next(c for c, lo, hi in LEN_CLASSES if lo <= length < hi)
        buckets.setdefault((n["midi"], lc), []).append(dict(n, length=length))
    index = []
    for (midi, lc), items in sorted(buckets.items()):
        items.sort(key=lambda c: -(c["voiced"] + 0.3 * np.log10(c["rms"])))
        for k, c in enumerate(items[:2]):
            name = f"{NOTES[midi % 12]}{midi // 12 - 1}_{lc}_{k:02d}.wav"
            s = cut(y, c["t"] - PRE, c["t"] + c["length"])
            s = s * (0.1 / max(float(np.sqrt(np.mean(s[:int(0.3 * SR)] ** 2))), 1e-6))
            s = s * min(1.0, 0.95 / max(float(np.abs(s).max()), 1e-6))
            sf.write(os.path.join(out, "bass", name), s, SR)
            index.append(dict(file=f"bass/{name}", kind="bass", midi=midi, len_class=lc, dur=len(s) / SR))
    return index


def ensure_bass(ref, out):
    """Adds the bass notes to a bank track made before they existed (drums and guitar, and the user's deletions, stay)."""
    path = os.path.join(out, "index.json")
    idx = json.load(open(path))
    if any(s["kind"] == "bass" for s in idx["samples"]) or not os.path.exists(os.path.join(ref, "bass.wav")):
        return False
    shutil.rmtree(os.path.join(out, "bass"), ignore_errors=True)
    os.makedirs(os.path.join(out, "bass"))
    bass = build_bass(ref, out)
    idx["samples"] += bass
    json.dump(idx, open(path, "w"), indent=1)
    print(f"{idx['track']}: bass notes {len(bass)} ({len({s['midi'] for s in bass})} pitches)")
    return True


def build(ref, bank, redo=False):
    name = os.path.basename(os.path.normpath(ref))
    out = os.path.join(bank, name)
    if os.path.exists(os.path.join(out, "index.json")) and not redo:
        if not ensure_bass(ref, out):
            print(f"{name}: already in the bank (use --redo)")
        return
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("drums", "guitar", "bass"):
        os.makedirs(os.path.join(out, sub))
    drums = build_drums(ref, out)
    guitar, n_cand = build_guitar(ref, out)
    bass = build_bass(ref, out)
    json.dump(dict(track=name, samples=drums + guitar + bass), open(os.path.join(out, "index.json"), "w"), indent=1)
    cnt = {k: sum(s["kind"] == k for s in drums) for k in KEEP_DRUMS}
    print(f"{name}: drums {cnt}, guitar slices {len(guitar)} of {n_cand} clean chords "
          f"({len({s['root'] for s in guitar})}/12 roots), bass notes {len(bass)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+", help="<root>/reference/<name> folders")
    ap.add_argument("--bank", default=None, help="default: <root>/native/bank")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    refs = []
    for r in args.refs:
        r = os.path.abspath(r)
        if os.path.exists(os.path.join(r, "drums.wav")):
            refs.append(r)
        else:                                       # a `reference` folder: every track in it
            refs += sorted(os.path.join(r, d) for d in os.listdir(r) if os.path.exists(os.path.join(r, d, "drums.wav")))
    for ref in refs:
        root = os.path.dirname(os.path.dirname(ref))
        build(ref, args.bank or os.path.join(root, "native", "bank"), args.redo)


if __name__ == "__main__":
    main()
