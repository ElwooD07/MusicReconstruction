"""UI of route B: replay the song with REAL samples cut from reference tracks, and listen at once.

Run through the launcher (main env):
    run.bat ui [--port 7860] [--no-browser]

Working folder ("root") = a folder named after the song, next to it (<dir>/<song>/ for <dir>/<song>.mp3; a song
that already lies in a folder of its own name uses that folder). The song file itself is never moved or copied:
    <root>/orig/        all stems of the song (made automatically, once)
    <root>/reference/   all stems of every reference track (made automatically, once)
    <root>/native/bank/ samples cut from the references: drums/*.wav, guitar/*.wav (listen, delete the bad ones)
    <root>/native/out/  the arrangements
The work itself is in tools/native_bank.py, tools/native_song.py, tools/native_arrange.py (also usable without the UI).
"""
import argparse
import contextlib
import functools
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

import gradio as gr
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import native_arrange  # noqa: E402
import native_bank  # noqa: E402

UI_DIR = os.path.join(ROOT, "temp", "ui")        # scratch: player copies, the remembered song
STATE_FILE = os.path.join(UI_DIR, "state.json")
PLAY_DIR = os.path.join(UI_DIR, "play")
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".wma"}
STOP_JS = "() => document.querySelectorAll('audio').forEach(a => { a.pause(); a.currentTime = 0; })"
HEAD = """<script>
/* one sound at a time: starting any player pauses the others */
document.addEventListener('play', e => document.querySelectorAll('audio').forEach(a => { if (a !== e.target) a.pause(); }), true);
</script>"""
os.makedirs(UI_DIR, exist_ok=True)

state = {"base": None, "root": None, "name": None, "duration": 200.0}


# ----------------------------------------------------------------------------------------- files & players
def save_state(**kw):
    s = json.load(open(STATE_FILE, encoding="utf-8")) if os.path.exists(STATE_FILE) else {}
    s.update(kw)
    json.dump(s, open(STATE_FILE, "w", encoding="utf-8"))


def slug(path):
    return re.sub(r"[^a-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0].lower()).strip("_") or "track"


def song_root(path):
    """Working folder of a song: <its dir>/<song name>/, or its own dir when that is already named after the song."""
    path = os.path.abspath(path)
    parent, stem = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
    if os.path.normcase(os.path.basename(parent)) == os.path.normcase(stem):
        return parent
    return os.path.join(parent, stem)


def ref_dir(name):
    return os.path.join(state["root"], "reference", name)


def bank_dir():
    return os.path.join(state["root"], "native", "bank")


def demux(audio, out_dir):
    """All stems (demucs lives in this env); a finished folder is reused."""
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "demux.py"), audio, out_dir],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise gr.Error(f"demux failed: {r.stderr[-400:]}")


def friendly(fn):
    """Show the real error text in the UI (Gradio prints only a bare 'Error' for unexpected exceptions)."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except gr.Error:
            raise
        except SystemExit as e:                      # the tools report problems with sys.exit(text)
            raise gr.Error(str(e)) from e
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}") from e
    return wrapper


def need_song():
    if not state["root"]:
        raise gr.Error("Спочатку оберіть пісню.")


def playable(path):
    """Gradio only serves files from the project folder or the system temp: the player gets a copy."""
    os.makedirs(PLAY_DIR, exist_ok=True)
    out = os.path.join(PLAY_DIR, f"{int(time.time() * 1000)}_{os.path.basename(path)}")
    shutil.copyfile(path, out)
    for old in sorted(glob.glob(os.path.join(PLAY_DIR, "*.wav")))[:-16]:
        os.remove(old)
    return out


def to_player(path, label):
    """Outputs for [player, now-playing line]: the file goes into the player and starts at once."""
    return gr.update(value=playable(path), autoplay=True), f"▶ **{label}**"


def pick(title, folder=False):
    """Native file / folder dialog (the UI runs on this machine); '' when cancelled or when there is no tkinter."""
    try:
        from tkinter import Tk, filedialog
        tk = Tk()
        tk.withdraw()
        tk.attributes("-topmost", True)
        if folder:
            p = filedialog.askdirectory(title=title)
        else:
            p = filedialog.askopenfilename(title=title, filetypes=[("Audio", "*.mp3 *.wav *.flac *.wma *.m4a *.ogg"),
                                                                   ("All files", "*.*")])
        tk.destroy()
        return p or ""
    except Exception as e:  # no tkinter: the path can still be pasted
        gr.Warning(f"file dialog unavailable ({e}); paste the path instead")
        return ""


@friendly
def browse(title):
    return pick(title) or gr.update()


@friendly
def open_root():
    need_song()
    os.startfile(state["root"])


@friendly
def open_bank():
    need_song()
    os.makedirs(bank_dir(), exist_ok=True)
    os.startfile(bank_dir())


# ----------------------------------------------------------------------------------------- song
def reference_names():
    if not state["root"]:
        return []
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(state["root"], "reference", "*", "drums.wav")))


def bank_tracks():
    """Reference tracks that are in the bank."""
    if not state["root"]:
        return []
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(bank_dir(), "*", "index.json")))


def bank_samples():
    """[(label, path)] of the samples that are still in the bank folders."""
    out = []
    for t in bank_tracks():
        for s in json.load(open(os.path.join(bank_dir(), t, "index.json"), encoding="utf-8"))["samples"]:
            p = os.path.join(bank_dir(), t, s["file"])
            if os.path.exists(p):
                what = s["kind"] if s["kind"] != "guitar" else f"гітара {native_bank.NOTES[s['root']]} {s['len_class']}"
                out.append((f"{t[:24]} · {what} · {os.path.basename(p)}", p))
    return out


def outputs():
    if not state["root"]:
        return []
    files = sorted(glob.glob(os.path.join(state["root"], "native", "out", "*.wav")), key=os.path.getmtime, reverse=True)
    return [(os.path.basename(f), f) for f in files]


def song_updates():
    """Values for every control that depends on the loaded song (see `song_out` in build())."""
    d = int(state["duration"])
    refs, tracks, outs = reference_names(), bank_tracks(), outputs()
    best_d, best_g = native_arrange.best_tracks(state["root"]) if tracks else (None, None)
    return (state["root"] or "", gr.update(value=map_rows()),
            gr.update(maximum=d, value=min(40, max(d - 20, 0))),
            gr.update(choices=refs, value=refs), gr.update(choices=refs, value=None),
            gr.update(choices=tracks, value=best_d), gr.update(choices=tracks, value=best_g),
            gr.update(choices=tracks, value=None), gr.update(choices=tracks, value=None),
            gr.update(choices=bank_samples(), value=None),
            gr.update(choices=outs, value=outs[0][1] if outs else None), bank_status())


def bank_status():
    if not state["root"]:
        return ""
    n_ref, n_bank = len(reference_names()), len(bank_tracks())
    return f"Зразків зі стемами: {n_ref}; у банку: {n_bank}." if n_ref else "Зразків ще немає: додайте їх у блоці 2."


def set_song(path):
    root = song_root(path)
    state.update(base=path, root=root, name=os.path.splitext(os.path.basename(path))[0],
                 duration=sf.info(os.path.join(root, "orig", "full.wav")).duration)
    save_state(base=path)


@friendly
def load_song(path, progress=gr.Progress()):
    path = (path or "").strip().strip('"')
    if not os.path.isfile(path):
        raise gr.Error("Файл пісні не знайдено. Натисніть «Огляд…» або вставте повний шлях.")
    progress(0.1, desc="Розділення на стеми (один раз, ~1-2 хв)…")
    demux(path, os.path.join(song_root(path), "orig"))
    set_song(path)
    return song_updates()


def restore():
    """Re-open the last song without redoing the demux."""
    if os.path.exists(STATE_FILE):
        base = json.load(open(STATE_FILE, encoding="utf-8")).get("base")
        if base and os.path.exists(os.path.join(song_root(base), "orig", "full.wav")):
            set_song(base)


MAP_HEADERS = ["такт", "час", "акорд 1-ї половини", "акорд 2-ї половини", "енергія 0-2"]
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def parse_chord(text):
    """Root pitch class from 'A', 'c#', 'Bb'…"""
    t = str(text).strip().replace("♯", "#").replace("♭", "b").capitalize()
    t = FLATS.get(t, t)
    if t not in native_bank.NOTES:
        raise gr.Error(f"Невідомий акорд «{text}»: потрібна нота на кшталт A, C#, Bb.")
    return native_bank.NOTES.index(t)


def map_rows():
    """Rows of the song map (automatic guess + the user's corrections), [] without a song."""
    if not state["root"]:
        return []
    song = native_arrange.song_with_edits(state["root"])
    g, notes = song["grid"], native_bank.NOTES
    rows = []
    for b, lv in enumerate(song["level"]):
        t = g["bar0"] + b * g["bar"]
        rows.append([b, f"{int(t // 60)}:{t % 60:04.1f}", notes[song["roots"][2 * b]], notes[song["roots"][2 * b + 1]], lv])
    return rows


@friendly
def show_map(progress=gr.Progress()):
    """What the arranger took from the song (grid, chords, energy per bar), shown so that a wrong guess is visible."""
    need_song()
    progress(0.1, desc="Аналіз пісні…")
    song = native_arrange.analyse(state["root"])
    g = song["grid"]
    n_edit = sum(map(len, native_arrange.load_edits(state["root"]).values()))
    return (f"**Сітка:** {g['bpm']:.2f} BPM (збіг барабанів із сіткою {100 * g['coherence']:.0f}%), {song['n_bars']} тактів. "
            f"Виправлень користувача: {n_edit}.  \nГітари грають на коренях акордів із таблиці, малюнок залежить від енергії такту.",
            map_rows())


@friendly
def save_map(rows):
    """Store the cells that differ from the automatic guess; they are applied to every later render."""
    need_song()
    auto = native_arrange.analyse(state["root"])
    edits = {"roots": {}, "levels": {}}
    for r in rows:
        if r[0] in ("", None):
            continue
        b = int(r[0])
        for k, cell in ((2 * b, r[2]), (2 * b + 1, r[3])):
            pc = parse_chord(cell)
            if pc != auto["roots"][k]:
                edits["roots"][str(k)] = pc
        lv = int(float(r[4]))
        if lv not in (0, 1, 2):
            raise gr.Error(f"Такт {b}: енергія має бути 0, 1 або 2.")
        if lv != auto["level"][b]:
            edits["levels"][str(b)] = lv
    native_arrange.save_edits(state["root"], edits)
    gr.Info(f"Збережено виправлень: акордів {len(edits['roots'])}, тактів {len(edits['levels'])}.")
    return show_map.__wrapped__()


@friendly
def reset_map():
    need_song()
    if os.path.exists(native_arrange.edits_path(state["root"])):
        os.remove(native_arrange.edits_path(state["root"]))
    return show_map.__wrapped__()


@friendly
def map_select(evt: gr.SelectData):
    """A click on a row of the map puts the start of the fragment on that bar."""
    if not state["root"]:
        return gr.update()
    g = native_arrange.analyse(state["root"])["grid"]
    return gr.update(value=int(g["bar0"] + evt.index[0] * g["bar"]))


# ----------------------------------------------------------------------------------------- references
def slot_preview(name):
    """Player copy of a whole reference (its original mix as a small OGG, made once)."""
    src = os.path.join(ref_dir(name), "full.wav")
    os.makedirs(PLAY_DIR, exist_ok=True)
    out = os.path.join(PLAY_DIR, f"ref_{slug(state['root'])}_{name}.ogg")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        y, sr = sf.read(src, dtype="float32", always_2d=True)
        # writing a whole track in one call overflows the stack inside libsndfile's Vorbis encoder: write by seconds
        with sf.SoundFile(out, "w", sr, y.shape[1], format="OGG", subtype="VORBIS") as f:
            for i in range(0, len(y), sr):
                f.write(y[i:i + sr])
    return out


@friendly
def play_reference(name):
    if not name:
        return gr.update(), gr.update()
    return to_player(slot_preview(name), f"Зразок «{name}»")


def add_tracks(current, folder, progress):
    """Split new reference tracks into stems (into the song's folder) and tick them."""
    need_song()
    if folder:
        d = pick("Тека з треками-зразками", folder=True)
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in AUDIO_EXT) if d else []
    else:
        f = pick("Оберіть трек-зразок")
        files = [f] if f else []
    if not files:
        return gr.update(), gr.update(), bank_status()
    added = list(current)
    for i, f in enumerate(files):
        progress(i / len(files), desc=f"Стеми {i + 1}/{len(files)}: {os.path.basename(f)} (~1–2 хв на трек)")
        demux(f, ref_dir(slug(f)))
        added.append(slug(f))
    refs = reference_names()
    return gr.update(choices=refs, value=sorted(set(added))), gr.update(choices=refs), bank_status()


@friendly
def add_one(current, progress=gr.Progress()):
    return add_tracks(current, False, progress)


@friendly
def add_folder(current, progress=gr.Progress()):
    return add_tracks(current, True, progress)


# ----------------------------------------------------------------------------------------- bank
@friendly
def make_bank(selected, redo, progress=gr.Progress()):
    need_song()
    if not selected:
        raise gr.Error("Відмітьте зразки, з яких нарізати банк.")
    log = io.StringIO()
    for i, name in enumerate(selected):
        progress(i / len(selected), desc=f"Нарізання {i + 1}/{len(selected)}: {name}")
        with contextlib.redirect_stdout(log):
            native_bank.build(ref_dir(name), bank_dir(), redo)
    tracks = bank_tracks()
    best_d, best_g = native_arrange.best_tracks(state["root"])
    return (gr.update(choices=bank_samples(), value=None), gr.update(choices=tracks, value=best_d),
            gr.update(choices=tracks, value=best_g), gr.update(choices=tracks, value=None),
            gr.update(choices=tracks, value=None), "```\n" + log.getvalue() + "\n```")


@friendly
def play_sample(path):
    if not path or not os.path.exists(path):
        return gr.update(), gr.update()
    return to_player(path, os.path.relpath(path, bank_dir()))


@friendly
def delete_sample(path):
    if not path or not os.path.exists(path):
        gr.Warning("Спочатку оберіть семпл у списку: він і видалиться.")
        return gr.update(), gr.update()
    os.remove(path)
    return gr.update(choices=bank_samples(), value=None), f"Видалено: {os.path.relpath(path, bank_dir())}"


# ----------------------------------------------------------------------------------------- arrangement
def fragment(start, length, whole):
    """(start, length) in seconds for the arranger; length 0 = the whole song."""
    return (0.0, 0.0) if whole else (float(start), float(length))


def engine_opts(p0, p1, p2, human, max_shift, double, fills):
    return dict(patterns={0: p0, 1: p1, 2: p2}, human=float(human), max_shift=int(max_shift), double=bool(double),
                fills=bool(fills))


@friendly
def play_song(drum_track, guitar_track, start, length, whole, seed, vocals, bass, other, guitar, drums, crash, orig_drums,
              *opts, progress=gr.Progress()):
    need_song()
    if not drum_track or not guitar_track:
        raise gr.Error("Оберіть, з яких треків брати барабани й гітари (спершу нарізайте банк).")
    progress(0.1, desc="Гра семплами…")
    a, n = fragment(start, length, whole)
    out, text = native_arrange.arrange(state["root"], [drum_track], a, n, int(seed), "", vocals, bass, other, guitar,
                                       drums, crash, guitar_tracks=[guitar_track], orig_drums=orig_drums,
                                       **engine_opts(*opts))
    outs = outputs()
    return (*to_player(out, os.path.basename(out)), gr.update(choices=outs, value=out), text)


@friendly
def compare(part, start, length, whole, seed, *opts_and_progress):
    """The same fragment with every bank track in turn (drums or guitars only); the radio then plays them one click each."""
    need_song()
    *opts, progress = opts_and_progress
    tracks = bank_tracks()
    if not tracks:
        raise gr.Error("Банк порожній: спершу «Нарізати банк».")
    a, n = fragment(start, length, whole)
    paths = {}
    for i, t in enumerate(tracks):
        progress(i / len(tracks), desc=f"{'Барабани' if part == 'drums' else 'Гітари'}: {t}")
        paths.update(native_arrange.compare_tracks(state["root"], part, a, n or 20.0, int(seed), only=[t],
                                                           **engine_opts(*opts)))
    return gr.update(choices=list(paths), value=None), f"Готово: {len(paths)} варіантів, клік по треку грає його."


@friendly
def compare_drums(*a, progress=gr.Progress()):
    return compare("drums", *a, progress)


@friendly
def compare_guitar(*a, progress=gr.Progress()):
    return compare("guitar", *a, progress)


@friendly
def play_compared(part, track):
    """A click on a track in the comparison: plays it and makes it the choice for the drums / the guitars."""
    path = os.path.join(state["root"], "native", "compare", part, f"{track}.wav")
    if not track or not os.path.exists(path):
        return gr.update(), gr.update(), gr.update()
    label = ("Барабани" if part == "drums" else "Гітари") + f" «{track}»"
    return (*to_player(path, label), gr.update(value=track))


@friendly
def play_output(path):
    if not path or not os.path.exists(path):
        return gr.update(), gr.update()
    return to_player(path, os.path.basename(path))


# ----------------------------------------------------------------------------------------- UI
def build():
    restore()
    refs0, tracks0, outs0 = reference_names(), bank_tracks(), outputs()
    d0 = int(state["duration"])
    with gr.Blocks(title="Реконструкція музики: гра реальними семплами") as demo:
        gr.Markdown("**Як користуватись:** 1) пісня (поруч із нею з'явиться робоча тека); 2) треки-зразки: додайте, послухайте, "
                    "відмітьте, з яких брати звук; 3) «Нарізати банк» — з їхніх барабанів і гітар виходять окремі семпли, "
                    "послухайте їх і видаліть погані; 4) «Зіграти»: пісня переграється цими семплами, вокал і бас лишаються "
                    "оригінальними. Нічого не генерується: весь звук — справжні записи зразків.")
        gr.Markdown("### 1. Пісня")
        with gr.Row():
            base_path = gr.Textbox(state["base"] or "", label="Файл пісні", scale=5)
            base_browse = gr.Button("Огляд…", scale=1)
            base_load = gr.Button("Завантажити", variant="primary", scale=1)
        with gr.Row():
            root_box = gr.Textbox(state["root"] or "", label="Робоча тека (поруч із піснею, названа як вона)",
                                  interactive=False, scale=5)
            open_btn = gr.Button("Відкрити теку", scale=1)
        with gr.Row():
            analyse_btn = gr.Button("Що програма розуміє про пісню", scale=1)
            analysis = gr.Markdown("Сітка й карта тактів (акорди та енергія); нижче її можна виправити.")
        with gr.Accordion("Карта пісні: акорди й енергія по тактах (виправляйте хибне)", open=False):
            gr.Markdown("Програма вгадала корінь акорду для кожної половини такту й «енергію» такту (0 тиха, 1 середня, 2 гучна): "
                        "від неї залежать малюнок гітари й креш. Гітари грають рівно ці акорди, тож хибний корінь = фальш. "
                        "Виправте клітинки (нота: A, C#, Bb…) і натисніть «Зберегти правки». Клік по рядку ставить початок "
                        "фрагмента на цей такт.")
            map_tbl = gr.Dataframe(map_rows(), headers=MAP_HEADERS, datatype=["number", "str", "str", "str", "number"],
                                   interactive=True, column_count=(5, "fixed"), static_columns=[0, 1], type="array",
                                   max_height=380)
            with gr.Row():
                map_save_btn = gr.Button("Зберегти правки", variant="primary")
                map_reset_btn = gr.Button("Скинути правки до автоматичних")

        gr.Markdown("### 2. Треки-зразки")
        with gr.Row():
            add_btn = gr.Button("＋ Додати трек…", scale=1)
            add_dir_btn = gr.Button("＋ Додати всі треки з теки…", scale=1)
        refs = gr.CheckboxGroup(refs0, value=refs0, label="Нарізати банк із",
                                info="Кожен трек розділяється на стеми один раз (~1–2 хв).")
        with gr.Row():
            listen_ref = gr.Dropdown(refs0, label="Послухати зразок цілим", scale=4)
            stop_btn = gr.Button("⏹ Стоп", scale=1, min_width=80)

        gr.Markdown("### 3. Банк семплів")
        with gr.Row():
            bank_btn = gr.Button("Нарізати банк із відмічених", variant="primary", scale=2)
            redo = gr.Checkbox(False, label="Нарізати наново", info="Інакше вже нарізані треки пропускаються.", scale=1)
            bank_open = gr.Button("Відкрити теку банку", scale=1)
        bank_msg = gr.Markdown(bank_status())
        with gr.Row():
            sample = gr.Dropdown(bank_samples(), label="Семпл банку (клік грає)", scale=5,
                                 info="Барабани: kick, snare, hat, crash; гітари: акорд-«квінта» від ноти, довжина "
                                      "short/mid/long.")
            sample_del = gr.Button("Видалити семпл", scale=1)
        bank_now = gr.Markdown("")
        bank_player = gr.Audio(show_label=False, interactive=False, autoplay=True)

        gr.Markdown("### 4. Зіграти пісню")
        gr.Markdown("**Не змішуйте треки.** Барабани беруться з одного треку, гітари з одного: суміш різних комплектів "
                    "звучить кашею. Порівняйте треки на одному фрагменті (кнопки нижче), клік по треку в списку грає його й "
                    "одразу вибирає. Тоді «Зіграти».")
        with gr.Row():
            start = gr.Slider(0, d0, min(40, max(d0 - 20, 0)), step=1, label="Початок фрагмента, с")
            length = gr.Slider(5, 60, 20, step=1, label="Довжина фрагмента, с", info="Короткий фрагмент рахується за секунди.")
            whole = gr.Checkbox(False, label="Вся пісня", info="Ігнорує початок і довжину.")
            seed = gr.Number(1, precision=0, label="Сід",
                             info="Інший сід = інший вибір семплів і «людський» таймінг; решта однакова.")
        with gr.Row():
            cmp_drums_btn = gr.Button("Порівняти барабани всіх треків")
            cmp_guitar_btn = gr.Button("Порівняти гітари всіх треків")
        with gr.Row():
            cmp_drums = gr.Radio(tracks0, label="Барабани (з оригінальними вокалом і басом)", value=None)
            cmp_guitar = gr.Radio(tracks0, label="Гітари (з оригінальними вокалом, басом і барабанами)", value=None)
        cmp_msg = gr.Markdown("")
        best0 = native_arrange.best_tracks(state["root"]) if tracks0 else (None, None)
        with gr.Row():
            drum_track = gr.Dropdown(tracks0, value=best0[0], label="Барабани з",
                                     info="Спершу обрано найповніше оснащений трек: це лише початок, вирішує слух.")
            guitar_track = gr.Dropdown(tracks0, value=best0[1], label="Гітари з")
        gr.Markdown("Малюнок гітари за енергією такту (див. карту пісні):")
        pattern_choices = [(label, key) for key, (label, _, _) in native_arrange.PATTERN_LIBRARY.items()]
        with gr.Row():
            p_quiet = gr.Dropdown(pattern_choices, value=native_arrange.DEFAULT_PATTERNS[0], label="У тихих тактах (0)")
            p_mid = gr.Dropdown(pattern_choices, value=native_arrange.DEFAULT_PATTERNS[1], label="У середніх тактах (1)")
            p_loud = gr.Dropdown(pattern_choices, value=native_arrange.DEFAULT_PATTERNS[2], label="У гучних тактах (2)")
        with gr.Accordion("Додатково", open=False):
            with gr.Row():
                human = gr.Slider(0, 3, 1.0, step=0.1, label="«Людськість» таймінгу",
                                  info="0 = точно по сітці, як машина; 1 = типово; більше = недбаліше.")
                max_shift = gr.Slider(0, 6, 3, step=1, label="Найбільший зсув семпла гітари, півтонів",
                                      info="Менше = природніший тембр, але потрібні семпли поруч за висотою.")
                orig_drums = gr.Slider(0, 1.5, 0.0, step=0.05, label="Оригінальні барабани",
                                       info="0 = замінені семплами; можна лишити частково для склейки.")
            with gr.Row():
                double = gr.Checkbox(True, label="Дві гітари (дубль у панорамі)", info="Вимкнено = одна по центру.")
                fills = gr.Checkbox(False, label="Філи", info="Перед гучнішим тактом остання доля стає снейр-роллом.")
        with gr.Accordion("Рівні", open=False):
            with gr.Row():
                g_voc = gr.Slider(0, 2, 1.0, step=0.05, label="Вокал оригіналу")
                g_bas = gr.Slider(0, 2, 1.0, step=0.05, label="Бас оригіналу")
                g_oth = gr.Slider(0, 2, 0.0, step=0.05, label="Електроніка оригіналу",
                                  info="0 = прибрана: її місце займають гітари.")
            with gr.Row():
                g_gtr = gr.Slider(0, 2, 1.0, step=0.05, label="Гітари", info="1 = типовий баланс до вокалу.")
                g_drm = gr.Slider(0, 2, 1.0, step=0.05, label="Барабани", info="Оригінальні барабани замінено завжди.")
                crash = gr.Checkbox(True, label="Креш на початку гучнішого розділу")
        play_btn = gr.Button("Зіграти", variant="primary")
        summary = gr.Markdown("")
        out_now = gr.Markdown("")
        out_player = gr.Audio(show_label=False, interactive=False, autoplay=True)
        history = gr.Dropdown(outs0, value=outs0[0][1] if outs0 else None, label="Попередні результати (нові зверху)")

        song_out = [root_box, map_tbl, start, refs, listen_ref, drum_track, guitar_track, cmp_drums, cmp_guitar, sample, history,
                    bank_msg]
        opts = [p_quiet, p_mid, p_loud, human, max_shift, double, fills]
        play_in = [drum_track, guitar_track, start, length, whole, seed, g_voc, g_bas, g_oth, g_gtr, g_drm, crash,
                   orig_drums, *opts]
        cmp_in = [start, length, whole, seed, *opts]
        base_browse.click(lambda: browse("Оберіть пісню"), None, base_path)
        base_load.click(load_song, base_path, song_out)
        open_btn.click(open_root)
        analyse_btn.click(show_map, None, [analysis, map_tbl])
        map_save_btn.click(save_map, map_tbl, [analysis, map_tbl])
        map_reset_btn.click(reset_map, None, [analysis, map_tbl])
        map_tbl.select(map_select, None, start)
        add_btn.click(add_one, refs, [refs, listen_ref, bank_msg])
        add_dir_btn.click(add_folder, refs, [refs, listen_ref, bank_msg])
        listen_ref.input(play_reference, listen_ref, [bank_player, bank_now])
        stop_btn.click(None, None, None, js=STOP_JS)
        bank_btn.click(make_bank, [refs, redo], [sample, drum_track, guitar_track, cmp_drums, cmp_guitar, bank_msg])
        bank_open.click(open_bank)
        sample.input(play_sample, sample, [bank_player, bank_now])
        sample_del.click(delete_sample, sample, [sample, bank_now])
        cmp_drums_btn.click(compare_drums, cmp_in, [cmp_drums, cmp_msg])
        cmp_guitar_btn.click(compare_guitar, cmp_in, [cmp_guitar, cmp_msg])
        cmp_drums.input(lambda t: play_compared("drums", t), cmp_drums, [out_player, out_now, drum_track])
        cmp_guitar.input(lambda t: play_compared("guitar", t), cmp_guitar, [out_player, out_now, guitar_track])
        play_btn.click(play_song, play_in, [out_player, out_now, history, summary])
        history.input(play_output, history, [out_player, out_now])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    build().queue().launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.no_browser, show_error=True,
                           head=HEAD)
