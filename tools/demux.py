"""Split any audio file into ALL stems, the same way for the song and for a reference.

Usage (main env):
    python tools/demux.py <audio-file> <out-dir>
Writes <out-dir>/{full,vocals,no_vocals,drums,bass,other}.wav (44.1 kHz stereo):
  1. htdemucs_ft, two stems -> clean vocals + no_vocals
  2. htdemucs on no_vocals  -> drums, bass, other
A finished <out-dir> is left untouched (delete it to redo).
"""
import argparse
import os
import shutil
import subprocess
import sys

import av
import numpy as np
import soundfile as sf

STEMS = ["full", "vocals", "no_vocals", "drums", "bass", "other"]


def decode(path, sr=44100):
    c = av.open(path)
    rs = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
    out = []
    for fr in c.decode(c.streams.audio[0]):
        out += [r.to_ndarray() for r in rs.resample(fr)]
    out += [r.to_ndarray() for r in rs.resample(None)]
    y = np.concatenate(out, axis=1).T
    return y / max(1.0, float(np.abs(y).max()))


def demucs(args):
    subprocess.run([sys.executable, "-m", "demucs", "--device", "cuda", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    out = os.path.abspath(args.out_dir)
    if all(os.path.exists(os.path.join(out, f"{s}.wav")) for s in STEMS):
        return
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "_tmp")
    full = os.path.join(out, "full.wav")
    sf.write(full, decode(args.audio), 44100)
    demucs(["-n", "htdemucs_ft", "--two-stems", "vocals", "-o", tmp, full])
    ft = os.path.join(tmp, "htdemucs_ft", "full")
    demucs(["-n", "htdemucs", "-o", tmp, os.path.join(ft, "no_vocals.wav")])
    four = os.path.join(tmp, "htdemucs", "no_vocals")
    for s in ("vocals", "no_vocals"):
        shutil.move(os.path.join(ft, f"{s}.wav"), os.path.join(out, f"{s}.wav"))
    for s in ("drums", "bass", "other"):
        shutil.move(os.path.join(four, f"{s}.wav"), os.path.join(out, f"{s}.wav"))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
