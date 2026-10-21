"""Check a pack of reference tracks against the requirements in PLAN.md; optionally split each into stems.

Usage (main env):
    python tools/check_tracks.py <folder> [--demux] [--out <dir>]
Scans the audio files directly inside <folder> and prints one line per track (format, bitrate, length, BPM,
tempo drift, key) plus a verdict on the whole pack. Writes <folder>/tracks_report.json and, if missing, a
notes.txt template. With --demux every track is also split into <out>/reference/<name>/{full,vocals,no_vocals,
drums,bass,other}.wav (<out> defaults to <folder>; <name> is the same slug the UI uses). Copy that `reference`
folder into the song's root folder and the UI sees the tracks as references without separating them again.
Metrics are only a hint: the final judgement of a track is by ear.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import av
import librosa
import numpy as np

from analyze_structure import key_of
from demux import decode

SR = 22050
EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus"}
LOSSLESS = ("flac", "alac", "pcm_")
BPM_RANGE = (130, 155)
MIN_KBPS = 192
MAX_DRIFT = 0.03          # (fastest - slowest third of the track) / mean tempo
NOTES_TEMPLATE = """# 3-5 reference tracks (file names) and what to copy from each (guitar tone, drum sound, bass...).
# Free text; one line per track:  <file name> - <what to copy>
"""


def slug(path):
    """Same as slug() in ui.py: the reference folder name the UI looks for."""
    return re.sub(r"[^a-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0].lower()).strip("_") or "track"


def analyze(path):
    with av.open(path) as c:
        s = c.streams.audio[0]
        codec, kbps = s.codec_context.name, (s.bit_rate or c.bit_rate or 0) / 1000
        duration = (c.duration or 0) / 1e6
    y = decode(path, SR).mean(1)
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, units="time", start_bpm=140)
    tempo = float(np.atleast_1d(tempo)[0])
    thirds = [60 / np.median(np.diff(b)) for b in np.array_split(beats, 3) if len(b) > 2]
    drift = (max(thirds) - min(thirds)) / np.mean(thirds) if len(thirds) > 1 else 1.0
    key, key_corr = key_of(librosa.feature.chroma_cqt(y=y, sr=SR).mean(1))
    return dict(file=os.path.basename(path), codec=codec, kbps=round(kbps), lossless=codec.startswith(LOSSLESS),
                duration=duration, bpm=tempo, drift=float(drift), key=key, key_corr=key_corr)


def in_range(bpm):
    return BPM_RANGE[0] <= bpm <= BPM_RANGE[1]


def problems(t):
    out = []
    if not t["lossless"] and t["kbps"] < MIN_KBPS:
        out.append(f"bitrate {t['kbps']} < {MIN_KBPS} kbps")
    if not in_range(t["bpm"]):
        half = [m for m in (t["bpm"] * 2, t["bpm"] / 2) if in_range(m)]
        out.append(f"BPM {t['bpm']:.0f} outside {BPM_RANGE[0]}-{BPM_RANGE[1]}" +
                   (f" (the beat tracker may be off by an octave: {half[0]:.0f}?)" if half else ""))
    if t["drift"] > MAX_DRIFT:
        out.append(f"tempo drifts {100 * t['drift']:.0f}% (not a steady grid?)")
    return out


def verdict(tracks):
    n, minutes = len(tracks), sum(t["duration"] for t in tracks) / 60
    fast = sum(in_range(t["bpm"]) for t in tracks)
    minor = sum(t["key"].endswith("min") for t in tracks)
    rows = [(n >= 8, f"{n} tracks (at least 8, best 12-20)"),
            (minutes >= 30, f"{minutes:.0f} min in total (at least 30)"),
            (fast >= 5, f"{fast} tracks at {BPM_RANGE[0]}-{BPM_RANGE[1]} BPM (at least 5)"),
            (minor >= n / 2, f"{minor} of {n} tracks in a minor key (rough: major/minor of a full mix is often confused)")]
    return [dict(ok=ok, text=text) for ok, text in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--demux", action="store_true", help="also split every track into stems")
    ap.add_argument("--out", default=None, help="where `reference\\<name>\\` is written (default: <folder>)")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in EXTS)
    if not files:
        sys.exit(f"No audio files ({', '.join(sorted(EXTS))}) directly inside {folder}")
    names = [slug(f) for f in files]
    for d in {n for n in names if names.count(n) > 1}:
        print(f"WARNING: several files give the same reference name '{d}': only one of them can be used")

    tracks = []
    print(f"{'file':<44} {'codec':>7} {'kbps':>5} {'min':>5} {'BPM':>5} {'drift':>6} {'key':>6}")
    for f in files:
        t = analyze(f)
        t["problems"] = problems(t)
        tracks.append(t)
        print(f"{t['file'][:44]:<44} {t['codec']:>7} {t['kbps']:>5} {t['duration'] / 60:>5.1f} {t['bpm']:>5.0f} "
              f"{100 * t['drift']:>5.1f}% {t['key']:>6}")
        for p in t["problems"]:
            print(f"    ! {p}")

    checks = verdict(tracks)
    print("\nPack:")
    for c in checks:
        print(f"  [{'ok' if c['ok'] else '--'}] {c['text']}")
    print("Not checkable here (listen): one style, live mix, no audience or lo-fi, kick/snare/hats and guitars clear.")
    json.dump(dict(tracks=tracks, pack=checks), open(os.path.join(folder, "tracks_report.json"), "w",
              encoding="utf-8"), indent=1, ensure_ascii=False)
    notes = os.path.join(folder, "notes.txt")
    if not os.path.exists(notes):
        open(notes, "w", encoding="utf-8").write(NOTES_TEMPLATE)

    if args.demux:
        out = os.path.join(os.path.abspath(args.out or folder), "reference")
        for i, (f, name) in enumerate(zip(files, names), 1):
            print(f"[{i}/{len(files)}] stems: {name}", flush=True)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "demux.py"), f,
                            os.path.join(out, name)], check=True)
        print(f"Stems: {out}")


if __name__ == "__main__":
    main()
