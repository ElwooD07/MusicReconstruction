"""Per-bar map of the song: grid (native_song.fit_grid), key, vocal activity/long notes, guitar (other) activity, drums.

Usage (main env):
    python tools/analyze_structure.py <song-root-folder> [--out <root>/orig/bars.json]
<song-root-folder> is the folder UI/demux made (it holds orig/{vocals,drums,bass,other}.wav).
Prints one line per bar and writes the table as JSON (grid, key, vocal activity, guitar/drum density).
"""
import argparse
import json
import os

import librosa
import numpy as np
import soundfile as sf

from native_bank import load as load44
from native_song import fit_grid

SR = 22050
MIN_NOTE_S = 0.4
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load(path):
    y, s = sf.read(path, dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    return librosa.resample(y, orig_sr=s, target_sr=SR) if s != SR else y


def key_of(chroma):
    best = (-2, "")
    for k in range(12):
        for prof, name in ((MAJ, "maj"), (MIN, "min")):
            c = np.corrcoef(np.roll(prof, k), chroma)[0, 1]
            if c > best[0]:
                best = (c, f"{NOTES[k]}{name}")
    return best[1], float(best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="song root folder (contains orig/ with the stems made by tools/demux.py)")
    ap.add_argument("--out", default=None, help="default: <root>/orig/bars.json")
    args = ap.parse_args()
    stems = os.path.join(args.root, "orig")
    args.out = args.out or os.path.join(stems, "bars.json")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    drums, other, bass = (load(os.path.join(stems, f"{n}.wav")) for n in ("drums", "other", "bass"))
    voc = load(os.path.join(stems, "vocals.wav"))

    # the grid comes from native_song.fit_grid (44.1 kHz): a beat tracker is a few percent off on programmed music
    grid = fit_grid(*(load44(os.path.join(stems, f"{n}.wav")).mean(1) for n in ("drums", "bass", "other")))
    tempo, bar, grid0 = grid["bpm"], grid["bar"], grid["bar0"]
    n_bars = int((len(drums) / SR - grid0) // bar)
    print(f"tempo {tempo:.2f} BPM (sixteenth grid fits {100 * grid['coherence']:.0f}% of the drum hits), "
          f"bar = {bar:.3f} s, first downbeat {grid0:.2f} s, {n_bars} bars")

    f0, voiced, _ = librosa.pyin(voc, fmin=80, fmax=800, sr=SR, frame_length=2048, hop_length=512)
    hop_s = 512 / SR
    midi = np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=100.0)), np.nan)
    voc_rms = librosa.feature.rms(y=voc, hop_length=512)[0]
    gate = voc_rms > 0.15 * np.percentile(voc_rms, 90)
    chroma_all = librosa.feature.chroma_cqt(y=other + bass, sr=SR, hop_length=512)
    other_rms = librosa.feature.rms(y=other, hop_length=512)[0]
    o_on = librosa.onset.onset_detect(y=other, sr=SR, hop_length=512, units="time")
    d_on = librosa.onset.onset_detect(y=drums, sr=SR, hop_length=512, units="time")

    # held vocal notes (voiced, pitch within 1 semitone of the note start) of at least MIN_NOTE_S seconds
    long_notes, run_start, ref = [], None, None
    for i, m in enumerate(list(midi) + [np.nan]):
        if not np.isnan(m) and ref is not None and abs(m - ref) < 1.0:
            continue
        if run_start is not None and (i - run_start) * hop_s >= MIN_NOTE_S:
            long_notes.append((run_start * hop_s, i * hop_s))
        run_start, ref = (i, m) if not np.isnan(m) else (None, None)

    rows = []
    print(f"{'bar':>3} {'t':>6} {'key':>7} {'voc%':>5} {'longnote_s':>10} {'gtr_on':>6} {'gtr_dB':>7} {'drm_on':>6}")
    for b in range(n_bars):
        a, z = grid0 + b * bar, grid0 + (b + 1) * bar
        fa, fz = int(a / hop_s), int(z / hop_s)
        v = gate[fa:fz]
        voc_pct = 100 * float(v.mean()) if len(v) else 0.0
        longest = max([min(e, z) - max(st, a) for st, e in long_notes if st < z and e > a] + [0.0])
        key, kc = key_of(chroma_all[:, fa:fz].mean(1))
        g_on = int(((o_on >= a) & (o_on < z)).sum())
        d_on_n = int(((d_on >= a) & (d_on < z)).sum())
        g_db = 20 * np.log10(other_rms[fa:fz].mean() + 1e-6)
        rows.append(dict(bar=b, t0=a, t1=z, key=key, key_corr=kc, voc_pct=voc_pct,
                         longnote_s=longest, gtr_onsets=g_on, gtr_db=float(g_db), drum_onsets=d_on_n))
        print(f"{b:>3} {a:>6.1f} {key:>7} {voc_pct:>5.0f} {longest:>10.2f} {g_on:>6} {g_db:>7.1f} {d_on_n:>6}")
    json.dump(dict(tempo=tempo, bar=bar, grid0=grid0, long_notes=long_notes, bars=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
