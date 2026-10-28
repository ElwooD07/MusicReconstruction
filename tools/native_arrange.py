"""Route B: play the song again with REAL samples (drums and guitars sliced from reference tracks), nothing generated.

Usage (main env):
    python tools/native_arrange.py <song-root> [--tracks a,b] [--guitar-tracks c] [--start S --length L] [--seed N] [--tag name]
        [--vocals 1] [--bass 1] [--keep-other 0] [--guitar 1] [--drums 1] [--no-crash] [--fills] [--single] [--human 1]
        [--max-shift 3] [--pattern-quiet long] [--pattern-middle eighths_kick] [--pattern-loud sync]
<song-root> holds orig/{vocals,bass,drums,other}.wav (demux) and native/bank/<track>/ (tools/native_bank.py).
The song supplies the timing and the harmony, the bank supplies the sound:
  * grid: sixteenth-note grid fitted to the song's drums (native_song.fit_grid);
  * drums: the song's own kick/snare/hat pattern replayed with kit samples (round robin, velocity, human timing);
  * guitars: power chords on the chord roots of the song (bass + other stems), two double-tracked layers panned apart;
    the part follows the energy of the song's own backing layer per bar (sparse / palm-muted eighths / open chords);
    samples are pitch-shifted by at most 3 semitones, like a sampler;
  * a crash on the bar where the energy goes up;
  * vocals and bass stay original (levels are arguments); the original drums are replaced; `other` (the electronic
    layer) is replaced too unless --keep-other.
--tracks / --guitar-tracks pick the reference tracks that supply the drums / the guitars. Default: ONE track each, the
best equipped one in the bank (a pool of many kits and guitars sounds like mush); `all` pools every track.
--start and --length render only a fragment (fast): the way to audition. `--compare drums|guitar` renders the same
fragment with every track in turn. Writes `<song-root>/native/out/<song>-native-<tag>.wav`. Judge by ear; the printed
numbers are only levels.
"""
import argparse
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf

import native_song as ns
from native_bank import NOTES, SR, bass_notes, load

ANALYSIS_VERSION = 4


def tr_noop(text):
    """Marks a UI-visible string for the translation extractor (i18n.py); returns it unchanged. The UI translates it."""
    return text


MAX_SHIFT = 3                                       # semitones
MAX_BASS_SHIFT = 5                                  # semitones, for bass notes (few samples per octave)
LAYERS = ((1.0, 0.3, 0.0), (0.3, 1.0, 0.007))       # (left gain, right gain, delay s) of the two double-tracked layers
# Guitar patterns: id -> (label, [(step in the bar, kind)], lock to the kick). Kinds: m = palm-muted chop, o = open
# chord, a = accent. With the lock, chops are added wherever the song's kick falls between the listed steps.
EIGHTHS = [(0, "o")] + [(s, "m") for s in range(2, 16, 2)] + [(15, "m")]
PATTERN_LIBRARY = {
    "none": (tr_noop("no guitar"), [], False),
    "one": (tr_noop("one open chord on the downbeat"), [(0, "o")], False),
    "long": (tr_noop("long chords: beats 1 and 3"), [(0, "o"), (8, "o")], False),
    "quarters": (tr_noop("open chords on the quarter notes"), [(0, "o"), (4, "o"), (8, "o"), (12, "o")], False),
    "sync": (tr_noop("open quarter notes and syncopation"), [(0, "o"), (4, "o"), (8, "o"), (10, "a"), (12, "o"), (14, "a")], False),
    "eighths": (tr_noop("palm-muted eighths"), EIGHTHS, False),
    "eighths_kick": (tr_noop("palm-muted eighths + hits with the kick"), EIGHTHS, True),
    "sixteenths": (tr_noop("palm-muted sixteenths"), [(0, "o")] + [(s, "m") for s in range(1, 16)], False),
    "kick": (tr_noop("only with the kick"), [(0, "o")], True),
}
DEFAULT_PATTERNS = {0: "long", 1: "eighths_kick", 2: "sync"}      # by energy level of the bar: quiet / middle / loud
LEVEL_GAIN = {0: 0.8, 1: 0.9, 2: 1.0}
RHYTHM_GAIN = {0: 0.7, 1: 0.85, 2: 1.0}               # drums and bass by section: lower in the gaps, lift in the chorus (about -3 / -1.4 / 0 dB)
HUMAN = {"kick": (0.003, 0.0), "snare": (0.004, 0.004), "hat": (0.005, 0.0), "crash": (0.0, 0.0)}   # sigma, mean delay (s)
DRUM_GAIN = {"kick": 1.0, "snare": 1.0, "hat": 0.5, "crash": 0.7}


def stem(root, name, a=0.0, z=0.0):
    """Stem `name` of the song (stereo, SR); only [a, z) seconds when z is given."""
    path = os.path.join(root, "orig", f"{name}.wav")
    if z and sf.info(path).samplerate == SR:
        y, _ = sf.read(path, start=int(a * SR), stop=int(z * SR), dtype="float32", always_2d=True)
        return y
    y = load(path)
    return y[int(a * SR):int(z * SR)] if z else y


def analyse(root):
    """Grid, drum hits, chord roots and energy level per bar; cached in <root>/native/song.json."""
    cache = os.path.join(root, "native", "song.json")
    drums_path = os.path.join(root, "orig", "drums.wav")
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("version") == ANALYSIS_VERSION and c.get("mtime") == os.path.getmtime(drums_path):
            return c
    d, b, o, v = (stem(root, n).mean(1) for n in ("drums", "bass", "other", "vocals"))
    grid = ns.fit_grid(d, b, o)
    n_bars = int((len(d) / SR - grid["bar0"]) // grid["bar"]) + 1
    hits = ns.transcribe_drums(d, grid)
    roots, conf = ns.chord_roots(b, o, grid, n_bars)
    energy = ns.bar_energy(o, v, d, grid, n_bars)
    level = ns.section_levels(energy)
    out = dict(version=ANALYSIS_VERSION, mtime=os.path.getmtime(drums_path), grid=grid, n_bars=n_bars, hits=hits,
               roots=roots, level=[int(x) for x in level], energy=energy)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"), indent=1)
    return out


def edits_path(root):
    return os.path.join(root, "native", "edits.json")


def load_edits(root):
    """The user's corrections of the automatic song map: {'roots': {half-bar index: pitch class}, 'levels': {bar: level}}."""
    if os.path.exists(edits_path(root)):
        return json.load(open(edits_path(root), encoding="utf-8"))
    return {"roots": {}, "levels": {}}


def save_edits(root, edits):
    os.makedirs(os.path.dirname(edits_path(root)), exist_ok=True)
    json.dump(edits, open(edits_path(root), "w", encoding="utf-8"))


def song_with_edits(root):
    """analyse() with the user's corrections of chords and energy applied."""
    song = dict(analyse(root))
    e = load_edits(root)
    song["roots"] = [e["roots"].get(str(i), r) for i, r in enumerate(song["roots"])]
    song["level"] = [e["levels"].get(str(i), lv) for i, lv in enumerate(song["level"])]
    return song


def bank_tracks(root):
    base = os.path.join(root, "native", "bank")
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, "index.json")))


def best_tracks(root):
    """(drums, guitars) = the best equipped track of the bank for each: the most complete kit, the most chord roots.
    Only a starting point: the ear decides (see compare_tracks)."""
    base = os.path.join(root, "native", "bank")
    kit, gtr = {}, {}
    for t in bank_tracks(root):
        have = [s for s in json.load(open(os.path.join(base, t, "index.json")))["samples"]
                if os.path.exists(os.path.join(base, t, s["file"]))]
        cnt = {k: sum(s["kind"] == k for s in have) for k in ("kick", "snare", "hat", "crash")}
        kit[t] = min(cnt["kick"], 8) + min(cnt["snare"], 8) + min(cnt["hat"], 10) + min(cnt["crash"], 1)
        gtr[t] = (len({s["root"] for s in have if s["kind"] == "guitar"}), sum(s["kind"] == "guitar" for s in have))
    if not kit:
        sys.exit("The bank is empty: run tools/native_bank.py first")
    return max(kit, key=kit.get), max(gtr, key=gtr.get)


def resolve(root, tracks, guitar_tracks):
    """Track lists for the drums and for the guitars: explicit names, ['all'] = the whole bank, empty = the best one."""
    best_d, best_g = best_tracks(root)

    def pick(names, best):
        names = list(names)
        return bank_tracks(root) if names == ["all"] else names or [best]
    return pick(tracks, best_d), pick(guitar_tracks, best_g)


class Bank:
    """Samples of the chosen reference tracks: drums from `drum_tracks`, guitar slices from `guitar_tracks`."""

    def __init__(self, root, drum_tracks, guitar_tracks, bass_tracks=()):
        base = os.path.join(root, "native", "bank")
        have = bank_tracks(root)
        missing = [n for n in {*drum_tracks, *guitar_tracks, *bass_tracks} if n not in have]
        if missing or not have:
            sys.exit(f"Not in the bank {base}: {missing or 'everything'} (run tools/native_bank.py first)")
        self.drums, self.guitar, self.bass, self.cache, self.last = {}, [], [], {}, {}
        self.drum_names, self.guitar_names = drum_tracks, guitar_tracks
        self.max_shift = MAX_SHIFT
        for n in sorted({*drum_tracks, *guitar_tracks, *bass_tracks}):
            for s in json.load(open(os.path.join(base, n, "index.json")))["samples"]:
                path = os.path.join(base, n, s["file"])
                if not os.path.exists(path):             # deleted by the user: not wanted
                    continue
                s = dict(s, path=path, track=n)
                if s["kind"] == "guitar" and n in guitar_tracks:
                    self.guitar.append(s)
                elif s["kind"] == "bass":
                    if n in bass_tracks:
                        self.bass.append(s)
                elif s["kind"] != "guitar" and n in drum_tracks:
                    self.drums.setdefault(s["kind"], []).append(s)

    def audio(self, s, shift=0):
        key = (s["path"], shift)
        if key not in self.cache:
            y, _ = sf.read(s["path"], dtype="float32", always_2d=True)
            if s["kind"] in ("guitar", "bass"):
                y = y.mean(1, keepdims=True)
            if shift:
                ratio = 2 ** (shift / 12)
                y = librosa.resample(y.T, orig_sr=SR * ratio, target_sr=SR, res_type="soxr_hq").T
            peak = np.abs(y.mean(1))[:int(0.08 * SR)]
            self.cache[key] = (y, int(np.argmax(peak)))      # the hit is where the sample is loudest, not where it starts
        return self.cache[key]

    def drum(self, cls, rng):
        pool = self.drums.get(cls)
        if not pool:
            return None
        recent = self.last.setdefault(cls, [])
        choice = [s for s in pool if s["path"] not in recent] or pool
        s = choice[rng.integers(len(choice))]
        recent.append(s["path"])
        del recent[:max(0, len(recent) - min(3, len(pool) - 1))]
        return s

    def chord(self, root, classes, rng):
        """A guitar slice for the chord root: nearest roots first, the given length classes, not the last few used."""
        pool = []
        for s in self.guitar:
            if s["len_class"] not in classes:
                continue
            shift = ((root - s["root"] + 6) % 12) - 6
            if abs(shift) <= self.max_shift:
                pool.append((s, shift))
        if not pool:
            return None, 0
        recent = self.last.setdefault("guitar", [])
        w = np.array([np.exp(-0.7 * abs(sh)) * (0.1 if s["path"] in recent else 1.0) for s, sh in pool])
        s, shift = pool[rng.choice(len(pool), p=w / w.sum())]
        recent.append(s["path"])
        del recent[:-6]
        return s, shift


    def bassnote(self, midi, length, rng):
        """A bass sample for the note: the nearest pitch within MAX_BASS_SHIFT, preferring the length class of the note."""
        want = "short" if length < 0.22 else "mid" if length < 0.5 else "long"
        pool = [(s, midi - s["midi"]) for s in self.bass if abs(midi - s["midi"]) <= MAX_BASS_SHIFT]
        if not pool:
            return None, 0
        recent = self.last.setdefault("bass", [])
        w = np.array([np.exp(-0.6 * abs(sh)) * (1.0 if s["len_class"] == want else 0.15) * (0.3 if s["path"] in recent else 1.0)
                      for s, sh in pool])
        s, shift = pool[rng.choice(len(pool), p=w / w.sum())]
        recent.append(s["path"])
        del recent[:-4]
        return s, shift


def place(buf, y, att, t, gain, length=None):
    """Add sample `y` (frames, ch) to `buf` so that its loudest moment (index att) lands on time t."""
    a = int(round(t * SR)) - att
    if length is not None and len(y) > length:
        y = y[:length].copy()
        n = min(int(0.02 * SR), len(y))
        y[-n:] *= np.linspace(1, 0, n)[:, None]
    lo, hi = max(a, 0), min(a + len(y), len(buf))
    if hi <= lo:
        return
    seg = y[lo - a:hi - a]
    buf[lo:hi] += (seg if seg.shape[1] == buf.shape[1] else np.repeat(seg, buf.shape[1], 1)) * gain


def render_drums(song, bank, rng, n, crash, t0=0.0, human=1.0, fills=False):
    """Drum bus of n frames that starts at t0 seconds of the song (only the hits that reach the window are played).
    fills: in the bar before the energy goes up, the last beat becomes a snare roll."""
    g = song["grid"]
    lv = song["level"]
    buf = np.zeros((n, 2), dtype=np.float32)
    lo, hi = t0 - 1.0, t0 + n / SR + 0.2
    fill_bars = {b for b in range(len(lv) - 1) if fills and lv[b + 1] > lv[b]}
    hits = [h for h in song["hits"] if not (h[0] // 16 in fill_bars and h[0] % 16 >= 12)]
    for b in fill_bars:
        hits += [(b * 16 + 12 + i, "snare", 0.55 + 0.15 * i, 0.0) for i in range(4)]
    for k, cls, vel, _ in hits:
        if not lo <= g["bar0"] + k * g["step"] <= hi:
            continue
        sample = bank.drum(cls, rng)
        if sample is None:
            continue
        sigma, delay = HUMAN[cls]
        t = g["bar0"] + k * g["step"] + delay * human + rng.normal(0, sigma * human) - t0
        y, att = bank.audio(sample)
        base = 0.55 if cls != "hat" else 0.35
        section = RHYTHM_GAIN[lv[min(max(k // 16, 0), len(lv) - 1)]]
        place(buf, y, att, t, section * DRUM_GAIN[cls] * (base + (1 - base) * min(vel, 1.0)) * rng.uniform(0.93, 1.07))
    if crash:
        for b in range(1, len(lv)):
            if lv[b] > lv[b - 1] and lo <= g["bar0"] + b * g["bar"] <= hi:
                sample = bank.drum("crash", rng)
                if sample is not None:
                    y, att = bank.audio(sample)
                    place(buf, y, att, g["bar0"] + b * g["bar"] - t0, DRUM_GAIN["crash"])
    return buf


def bar_pattern(patterns, level, b, kick_steps):
    """[(step, kind)] the guitar plays in bar b at the given energy level."""
    _, steps, lock = PATTERN_LIBRARY[patterns[level]]
    pattern = list(steps)
    if lock:
        have = {s for s, _ in pattern}
        pattern += [(s, "m") for s in range(16) if (b * 16 + s) in kick_steps and s not in have]
        pattern.sort()
    return pattern


def render_guitars(song, bank, rng, n, t0=0.0, patterns=None, human=1.0, double=True):
    """Guitar bus of n frames that starts at t0 seconds of the song."""
    g = song["grid"]
    patterns = {**DEFAULT_PATTERNS, **(patterns or {})}
    layers = LAYERS if double else ((0.8, 0.8, 0.0),)
    kick_steps = {k for k, c, *_ in song["hits"] if c == "kick"}
    buf = np.zeros((n, 2), dtype=np.float32)
    for b, level in enumerate(song["level"]):
        if not t0 - 2.5 <= g["bar0"] + b * g["bar"] <= t0 + n / SR + 0.2:
            continue
        pattern = bar_pattern(patterns, level, b, kick_steps)
        for i, (step, kind) in enumerate(pattern):
            nxt = pattern[i + 1][0] if i + 1 < len(pattern) else 16
            length = int((nxt - step) * g["step"] * SR)
            root = song["roots"][min(2 * b + (1 if step >= 8 else 0), len(song["roots"]) - 1)]
            classes = ("short",) if kind == "m" else ("long", "mid") if nxt - step >= 6 else ("mid", "short")
            for left, right, delay in layers:
                sample, shift = bank.chord(root, classes, rng)
                if sample is None:
                    continue
                y, att = bank.audio(sample, shift)
                accent = 1.15 if step == 0 else 1.0
                gain = LEVEL_GAIN[level] * accent * rng.uniform(0.9, 1.1)
                t = g["bar0"] + (b * 16 + step) * g["step"] + delay + rng.normal(0, 0.004 * human) - t0
                place(buf, y, att, t, gain * np.array([left, right], dtype=np.float32), length + int(0.02 * SR))
    return buf


def song_bass(root):
    """Notes of the song's own (electronic) bass: [dict(t, length, midi, rms)]; cached in <root>/native/bass_notes.json."""
    path = os.path.join(root, "native", "bass_notes.json")
    src = os.path.join(root, "orig", "bass.wav")
    if os.path.exists(path):
        c = json.load(open(path))
        if c.get("mtime") == os.path.getmtime(src):
            return c["notes"]
    notes = [n for n in bass_notes(stem(root, "bass").mean(1)) if n["voiced"] >= 0.5 and n["rms"] > 0.005]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(mtime=os.path.getmtime(src), notes=notes), open(path, "w"))
    return notes


def render_bass(song, notes, bank, rng, n, t0=0.0, human=1.0):
    """Bass bus of n frames from t0: the song's own bass line (times, pitches, lengths) played with the bank's bass notes."""
    buf = np.zeros((n, 2), dtype=np.float32)
    top = np.percentile([x["rms"] for x in notes], 90) if notes else 1.0
    for x in notes:
        t = x["t"] - t0
        if not -1.0 <= t <= n / SR + 0.2:
            continue
        sample, shift = bank.bassnote(x["midi"], x["length"], rng)
        if sample is None:
            continue
        y, att = bank.audio(sample, shift)
        g = song["grid"]
        section = RHYTHM_GAIN[song["level"][min(max(int((x["t"] - g["bar0"]) // g["bar"]), 0), len(song["level"]) - 1)]]
        gain = section * (0.55 + 0.45 * min(x["rms"] / top, 1.0)) * rng.uniform(0.95, 1.05)
        place(buf, y, att, t + rng.normal(0, 0.003 * human), gain, int((min(x["length"], 1.5) + 0.03) * SR))
    return buf


def active_rms(y, frame=4096, floor=0.1):
    """RMS over the frames that are not near-silent (so that gaps do not decide the level)."""
    m = y.mean(1) if y.ndim > 1 else y
    r = np.array([np.sqrt(np.mean(m[i:i + frame] ** 2)) for i in range(0, len(m) - frame, frame)])
    r = r[r > floor * np.percentile(r, 90)] if len(r) else r
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else 1e-6


def arrange(root, tracks=(), start=0.0, length=0.0, seed=1, tag="", vocals=1.0, bass=1.0, keep_other=0.0, guitar=1.0,
            drums=1.0, crash=True, out=None, guitar_tracks=(), orig_drums=0.0, parts=("drums", "guitar"),
            patterns=None, human=1.0, max_shift=MAX_SHIFT, double=True, fills=False, bass_ref=False, bass_tracks=()):
    """Render the song (or `length` seconds from `start`) with the bank sound; returns (path of the wav, summary text).
    `tracks` / `guitar_tracks`: bank tracks for the drums / the guitars (empty = the best one, ['all'] = pooled).
    `parts` can leave out the drums or the guitars; `orig_drums` keeps the song's own drums (to audition guitars alone).
    `patterns` {energy level: id from PATTERN_LIBRARY}; `human` scales the timing looseness (0 = machine-exact);
    `max_shift` is the largest pitch shift of a guitar slice in semitones; `double` = two guitar layers; `fills` =
    snare roll before louder bars. `bass_ref`: the song's own bass is dropped and its bass LINE (transcribed) is played
    with bass notes of `bass_tracks` (empty = the guitar tracks). The song map is the automatic one with the user's
    corrections (edits.json)."""
    root = os.path.abspath(root)
    song = song_with_edits(root)
    g = song["grid"]
    total = sf.info(os.path.join(root, "orig", "vocals.wav")).duration
    a = min(max(start, 0.0), max(total - 1.0, 0.0))
    z = min(a + length, total) if length else total
    drum_tracks, gtr_tracks = resolve(root, tracks, guitar_tracks)
    bass_names = (list(bass_tracks) or gtr_tracks) if bass_ref else []
    bank = Bank(root, drum_tracks, gtr_tracks, bass_names)
    bank.max_shift = max_shift
    rng = np.random.default_rng(seed)
    voc, bass_stem, other, orig_d = (stem(root, n, a, z) for n in ("vocals", "bass", "other", "drums"))
    n = len(voc)
    v_rms = active_rms(voc)
    mix = vocals * voc + (0.0 if bass_ref else bass) * bass_stem + keep_other * other + orig_drums * orig_d
    if bass_ref:
        bass_bus = render_bass(song, song_bass(root), bank, rng, n, a, human)
        mix += bass_bus * (bass * 0.8 * v_rms / active_rms(bass_bus))
    if "drums" in parts:
        drum_bus = render_drums(song, bank, rng, n, crash, a, human, fills)
        mix += drum_bus * (drums * v_rms / active_rms(drum_bus))
    if "guitar" in parts:
        guitar_bus = render_guitars(song, bank, rng, n, a, patterns, human, double)
        mix += guitar_bus * (guitar * 0.8 * v_rms / active_rms(guitar_bus))
    peak = float(np.abs(mix).max())
    mix *= 0.97 / peak if peak > 0.97 else 1.0
    fade = min(int(0.05 * SR), len(mix) // 2)
    mix[:fade] *= np.linspace(0, 1, fade)[:, None]
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    if not tag:
        d = "all" if len(drum_tracks) > 2 else "-".join(t[:12] for t in drum_tracks)
        gt = "all" if len(gtr_tracks) > 2 else "-".join(t[:12] for t in gtr_tracks)
        tag = f"d-{d}_g-{gt}" + (f"_{int(a)}-{int(z)}s" if length else "")
    out = out or os.path.join(root, "native", "out", f"{os.path.basename(root)}-native-{tag}-s{seed}.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, mix, SR, subtype="PCM_16")
    lv = np.bincount(song["level"], minlength=3)
    drum_counts = {k: len(v) for k, v in bank.drums.items()}
    text = "\n".join([
        f"Fragment {a:.0f}-{z:.0f} s ({'whole song' if z - a >= total - 1 else f'{z - a:.0f} s'}), seed {seed}.",
        f"Drums from: {', '.join(drum_tracks)} {drum_counts}.",
        f"Guitars from: {', '.join(gtr_tracks)} ({len(bank.guitar)} slices).",
        *([f"Bass: the song's line, notes from {', '.join(bass_names)} ({len(bank.bass)} notes)."] if bass_ref else []),
        f"Grid {g['bpm']:.2f} BPM, {song['n_bars']} bars, sections nobody sings/verse/chorus = {lv.tolist()}."])
    return out, text


def compare_tracks(root, part, start, length, seed=1, only=None, **kw):
    """The same fragment with every track of the bank in turn, for the `part` ('drums' or 'guitar') only: drums are
    heard with the original vocals and bass, guitars with the original vocals, bass AND drums, so that only the part under
    test differs. Returns {track: wav path} (in <root>/native/compare/<part>/, overwritten on every call); `only` limits the tracks."""
    out = {}
    folder = os.path.join(os.path.abspath(root), "native", "compare", part)
    for t in only or bank_tracks(root):
        names = dict(tracks=[t]) if part == "drums" else dict(guitar_tracks=[t])
        path = os.path.join(folder, f"{t}.wav")
        arrange(root, start=start, length=length, seed=seed, out=path, parts=(part,),
                orig_drums=1.0 if part == "guitar" else 0.0, **names, **kw)
        out[t] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--tracks", default="", help="bank tracks for the drums, comma-separated; `all` = pooled; default: "
                                                 "the best one")
    ap.add_argument("--guitar-tracks", default="", help="bank tracks for the guitars (same rules)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--length", type=float, default=0.0, help="seconds from --start; 0 = to the end")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--vocals", type=float, default=1.0)
    ap.add_argument("--bass", type=float, default=1.0)
    ap.add_argument("--bass-ref", action="store_true",
                    help="drop the song's electronic bass, play its bass line with the bank's bass notes")
    ap.add_argument("--bass-tracks", default="", help="bank tracks for --bass-ref (default: the guitar tracks)")
    ap.add_argument("--keep-other", type=float, default=0.0)
    ap.add_argument("--guitar", type=float, default=1.0, help="guitar level relative to the default balance")
    ap.add_argument("--drums", type=float, default=1.0, help="drum level relative to the default balance")
    ap.add_argument("--no-crash", action="store_true")
    ap.add_argument("--fills", action="store_true", help="snare roll before louder bars")
    ap.add_argument("--single", action="store_true", help="one guitar layer instead of two double-tracked")
    ap.add_argument("--human", type=float, default=1.0, help="timing looseness, 0 = machine-exact")
    ap.add_argument("--max-shift", type=int, default=MAX_SHIFT, help="largest pitch shift of a guitar slice, semitones")
    for lv, name in ((0, "quiet"), (1, "middle"), (2, "loud")):
        ap.add_argument(f"--pattern-{name}", default=DEFAULT_PATTERNS[lv], choices=list(PATTERN_LIBRARY),
                        help=f"guitar pattern in {name} bars")
    ap.add_argument("--compare", choices=["drums", "guitar"], help="render the fragment once per bank track instead")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.compare:
        for t, p in compare_tracks(a.root, a.compare, a.start, a.length or 20.0, a.seed).items():
            print(f"{t}: {p}")
        return
    split = lambda v: [t for t in v.split(",") if t]
    out, text = arrange(a.root, split(a.tracks), a.start, a.length, a.seed, a.tag, a.vocals, a.bass, a.keep_other,
                        a.guitar, a.drums, not a.no_crash, a.out, split(a.guitar_tracks),
                        patterns={0: a.pattern_quiet, 1: a.pattern_middle, 2: a.pattern_loud}, human=a.human,
                        max_shift=a.max_shift, double=not a.single, fills=a.fills, bass_ref=a.bass_ref,
                        bass_tracks=split(a.bass_tracks))
    print(text)
    print(out)


if __name__ == "__main__":
    main()
