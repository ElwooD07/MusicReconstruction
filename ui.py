"""Tuning UI: pick a song, pick a reference sound, generate with ACE-Step, mix and listen at once.

Run through the launcher (ACE-Step venv, models on MODELS_DIR):
    run.bat ui [--port 7860] [--no-browser]

Working folder = the folder that holds the song you pick ("root"):
    <root>/orig/        all stems of the song (made automatically, once)
    <root>/reference/   all stems of every reference track (made automatically, once)
    <root>/gen/         every generated track with a JSON sidecar of its settings
    <root>/<song>-<date>-<time>.wav   the mixed results
"""
import argparse
import functools
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.abspath(__file__))
ACESTEP_DIR = os.path.abspath(os.environ.get("ACESTEP_DIR") or os.path.join(ROOT, "ace-step"))
sys.path.insert(0, ACESTEP_DIR)
os.environ.setdefault("ACESTEP_PROJECT_ROOT", ACESTEP_DIR)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.model_downloader import ensure_dit_model, ensure_main_model, get_checkpoints_dir  # noqa: E402

SR = 48000
UI_DIR = os.path.join(ROOT, "temp", "ui")        # scratch: context file, model output before it is copied
STATE_FILE = os.path.join(UI_DIR, "state.json")  # remembers the last song / reference
MAIN_PYTHON = os.path.join(ROOT, "env", "Scripts", "python.exe")
STEMS = ["vocals", "drums", "bass", "other"]
TRACKS = ["guitar", "keyboard", "synth", "strings", "percussion", "drums", "bass", "fx", "brass", "woodwinds"]
MODELS = ["acestep-v15-base", "acestep-v15-sft", "acestep-v15-turbo"]
REF_STEM_CHOICES = [("вокал", "vocals"), ("барабани", "drums"), ("бас", "bass"),
                    ("інше (гітари, клавіші)", "other"), ("весь мікс", "full")]
os.makedirs(UI_DIR, exist_ok=True)

_cache = {}
_engine = {"model": None, "handler": None, "llm": LLMHandler()}
state = {"base": None, "root": None, "name": None, "duration": 200.0}


# ----------------------------------------------------------------------------------------- files & audio
def save_state(**kw):
    s = json.load(open(STATE_FILE, encoding="utf-8")) if os.path.exists(STATE_FILE) else {}
    s.update(kw)
    json.dump(s, open(STATE_FILE, "w", encoding="utf-8"))


def slug(path):
    return re.sub(r"[^a-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0].lower()).strip("_") or "track"


def orig_dir():
    return os.path.join(state["root"], "orig")


def ref_dir(name):
    return os.path.join(state["root"], "reference", name)


def demux(audio, out_dir):
    """All stems via the main env (demucs lives there); a finished folder is reused."""
    r = subprocess.run([MAIN_PYTHON, os.path.join(ROOT, "tools", "demux.py"), audio, out_dir],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise gr.Error(f"demux failed: {r.stderr[-400:]}")


def load_audio(path, start=0.0, end=None, sr=SR):
    """Stereo float32 [n, 2] at `sr` for [start, end) seconds (the last resampled file is cached)."""
    key = (path, os.path.getmtime(path), sr)
    if key not in _cache:
        y, s = sf.read(path, dtype="float32", always_2d=True)
        y = np.repeat(y, 2, axis=1) if y.shape[1] == 1 else y[:, :2]
        if s != sr:
            y = torchaudio.functional.resample(torch.from_numpy(y.T), s, sr).T.numpy()
        _cache.clear()
        _cache[key] = y
    return _cache[key][int(start * sr):int(end * sr) if end else None]


def soft_limit(x, knee=0.8):
    a = np.abs(x)
    return np.where(a > knee, np.sign(x) * (knee + (1 - knee) * np.tanh((a - knee) / (1 - knee))), x).astype(np.float32)


def friendly(fn):
    """Show the real error text in the UI (Gradio prints only a bare 'Error' for unexpected exceptions)."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}") from e
    return wrapper


def playable(path):
    """Gradio only serves files from the project folder or the system temp: give the player a copy of files that
    live in the song's root folder (the files themselves stay where they are)."""
    play = os.path.join(UI_DIR, "play")
    os.makedirs(play, exist_ok=True)
    out = os.path.join(play, f"{int(time.time() * 1000)}_{os.path.basename(path)}")
    shutil.copyfile(path, out)
    for old in sorted(glob.glob(os.path.join(play, "*.wav")))[:-16]:  # keep the last 16 copies
        os.remove(old)
    return out


@friendly
def browse(title):
    """Native file dialog (the UI runs on this machine)."""
    try:
        from tkinter import Tk, filedialog
        tk = Tk()
        tk.withdraw()
        tk.attributes("-topmost", True)
        p = filedialog.askopenfilename(title=title, filetypes=[("Audio", "*.mp3 *.wav *.flac *.wma *.m4a *.ogg"),
                                                               ("All files", "*.*")])
        tk.destroy()
        return p or gr.update()
    except Exception as e:  # no tkinter: the path can still be pasted
        gr.Warning(f"file dialog unavailable ({e}); paste the path instead")
        return gr.update()


@friendly
def open_root():
    if not state["root"]:
        raise gr.Error("Спочатку оберіть пісню.")
    os.startfile(state["root"])


def reference_names():
    if not state["root"]:
        return []
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(state["root"], "reference", "*", "full.wav")))


def history():
    if not state["root"]:
        return []
    items = []
    for j in sorted(glob.glob(os.path.join(state["root"], "gen", "*.json")), reverse=True):
        m = json.load(open(j, encoding="utf-8"))
        items.append((f"{os.path.basename(j)[:-5]} | {m['task']} {m['track']} | сід {m['seed']} | кроки {m['steps']} | "
                      f"зразок: {m['ref'] or '—'} | {m['caption'][:35]}", j))
    return items


def song_status():
    if not state["base"]:
        return "Пісню не обрано."
    return f"Пісня: {os.path.basename(state['base'])} ({state['duration']:.0f} с). Стеми готові ✓"


def song_updates():
    """Values for every control that depends on the loaded song."""
    d = int(state["duration"])
    h = history()
    return (state["root"] or "—", song_status(),
            gr.update(maximum=d, value=min(40, d)), gr.update(maximum=d, value=min(100, d)),
            gr.update(choices=reference_names(), value=(reference_names() or [None])[0]),
            gr.update(choices=h, value=h[0][1] if h else None), gr.update(choices=h, value=h[1][1] if len(h) > 1 else None))


@friendly
def load_song(path, progress=gr.Progress()):
    path = (path or "").strip().strip('"')
    if not os.path.isfile(path):
        raise gr.Error("Файл пісні не знайдено. Натисніть «Огляд…» або вставте повний шлях.")
    progress(0.1, desc="Розділення на стеми (один раз, ~1-2 хв)…")
    root = os.path.dirname(os.path.abspath(path))
    demux(path, os.path.join(root, "orig"))
    state.update(base=path, root=root, name=os.path.splitext(os.path.basename(path))[0],
                 duration=sf.info(os.path.join(root, "orig", "full.wav")).duration)
    save_state(base=path)
    return song_updates()


@friendly
def load_reference(path, progress=gr.Progress()):
    if not state["root"]:
        raise gr.Error("Спочатку оберіть пісню: зразки зберігаються в її кореневій папці.")
    path = (path or "").strip().strip('"')
    if not os.path.isfile(path):
        raise gr.Error("Файл зразка не знайдено.")
    progress(0.1, desc="Розділення зразка на стеми…")
    name = slug(path)
    demux(path, ref_dir(name))
    return gr.update(choices=reference_names(), value=name), f"Зразок «{name}» готовий ✓"


def build_context(gains, start, end):
    parts = [load_audio(os.path.join(orig_dir(), f"{k}.wav"), start, end) * g for k, g in zip(STEMS, gains) if g > 0]
    if not parts:
        raise gr.Error("У контексті всі повзунки на нулі: моделі нічого «слухати».")
    n = min(len(p) for p in parts)
    ctx = sum(p[:n] for p in parts)
    ctx = ctx / max(1.0, float(np.abs(ctx).max()))
    path = os.path.join(UI_DIR, "ctx.wav")
    sf.write(path, ctx, SR)
    return path


def cut_reference(name, stems, start):
    """Exactly 30 s (looped if shorter) of the chosen reference stems from `start`; ACE-Step's own choice of
    3 random 10 s pieces is avoided."""
    mix = None
    for s in stems:
        y = load_audio(os.path.join(ref_dir(name), f"{s}.wav"))
        n = len(y) if mix is None else min(len(mix), len(y))
        mix = y[:n] if mix is None else mix[:n] + y[:n]
    need = 30 * SR
    mix = np.concatenate([mix] * int(np.ceil((start * SR + need) / len(mix))))
    seg = mix[int(start * SR):int(start * SR) + need]
    seg = seg / max(1e-6, float(np.abs(seg).max())) * 0.9
    path = os.path.join(UI_DIR, f"ref_{int(time.time() * 1000)}.wav")
    sf.write(path, seg, SR)
    return path


# ----------------------------------------------------------------------------------------- generation
# Free VRAM (GiB) at model-load time below which the model parts are offloaded to the CPU. Measured with a hard
# 8 GB cap: Lego of 60 s peaks at 7.8 GB without offload and at 5.1 GB with it (the same generation, 14 s vs 27 s).
OFFLOAD_BELOW_GIB = 7.5


def get_handler(model):
    if _engine["model"] != model:
        _engine["handler"] = None
        gc.collect()
        torch.cuda.empty_cache()
        free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
        offload = free_gib < OFFLOAD_BELOW_GIB
        if offload:
            gr.Info(f"Вільної відеопам'яті {free_gib:.1f} ГБ (< {OFFLOAD_BELOW_GIB}): модель працює з вивантаженням на CPU, "
                    f"повільніше, але вміщається.")
        ckpt = get_checkpoints_dir()
        ok, msg = ensure_main_model(checkpoints_dir=ckpt)
        if ok and model != "acestep-v15-turbo":
            ok, msg = ensure_dit_model(model, checkpoints_dir=ckpt)
        if not ok:
            raise gr.Error(f"модель {model}: {msg}")
        h = AceStepHandler()
        status, ok = h.initialize_service(project_root=ACESTEP_DIR, config_path=model, device="cuda",
                                          offload_to_cpu=offload, offload_dit_to_cpu=offload)
        if not ok:
            raise gr.Error(f"не вдалося запустити модель: {status}")
        _engine.update(model=model, handler=h, offload=offload)
    return _engine["handler"]


def generate(task, common, extra, progress):
    (model, caption, steps, guidance, seed, count, start, end, g_voc, g_drm, g_bas, g_oth,
     use_ref, ref_name, ref_stems, ref_start) = common
    if not state["root"]:
        raise gr.Error("Спочатку оберіть пісню (блок «1. Пісня»).")
    if end <= start:
        raise gr.Error("Кінець фрагмента має бути пізніше за початок.")
    if use_ref and not (ref_name and ref_stems):
        raise gr.Error("Увімкнено зразок, але не обрано зразок або його доріжки.")
    ctx = build_context([g_voc, g_drm, g_bas, g_oth], start, end)
    ref = cut_reference(ref_name, ref_stems, ref_start) if use_ref else None
    handler = get_handler(model)
    gen_dir = os.path.join(state["root"], "gen")
    os.makedirs(gen_dir, exist_ok=True)
    for i in range(int(count)):
        sd = int(seed) + i
        progress((i, int(count)), desc=f"сід {sd}")
        kw = dict(task_type=task, src_audio=ctx, caption=caption, lyrics="[Instrumental]",
                  instrumental=(task == "cover"), duration=-1, seed=sd, thinking=False,
                  inference_steps=int(steps), guidance_scale=float(guidance))
        if task == "lego":
            track, rep_start, rep_end = extra
            kw.update(instruction=f"Generate the {track} track based on the audio context:")
            if rep_start > 0 or rep_end != -1:
                kw.update(repainting_start=float(rep_start), repainting_end=float(rep_end))
        else:
            track = "cover"
            noise, strength = extra
            kw.update(audio_cover_strength=float(strength), cover_noise_strength=float(noise))
        if ref:
            kw.update(reference_audio=ref)
        t0 = time.time()
        res = generate_music(handler, _engine["llm"], GenerationParams(**kw),
                             GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False, seeds=[sd]),
                             save_dir=os.path.join(UI_DIR, "tmp"))
        if not res.success:
            raise gr.Error(f"сід {sd}: {getattr(res, 'error', res)}")
        gid = time.strftime("%Y%m%d-%H%M%S") + f"_s{sd}"
        out = os.path.join(gen_dir, f"{gid}.wav")
        shutil.copyfile(res.audios[0]["path"], out)
        meta = dict(task=task, track=track, seed=sd, steps=int(steps), guidance=guidance, caption=caption, start=start,
                    end=end, ref=(f"{ref_name}:{'+'.join(ref_stems)}@{ref_start:.0f}с" if ref else ""),
                    ctx_gains=[g_voc, g_drm, g_bas, g_oth], wav=out, model=model, took=round(time.time() - t0, 1),
                    root=state["root"], name=state["name"], extra=list(extra))
        json.dump(meta, open(os.path.join(gen_dir, f"{gid}.json"), "w", encoding="utf-8"), indent=1)
    h = history()
    return (playable(out), gr.update(choices=h, value=h[0][1]), gr.update(choices=h, value=h[1][1] if len(h) > 1 else None),
            ref, f"Готово: {int(count)} варіант(ів), останній за {meta['took']} с")


@friendly
def gen_lego(*a, progress=gr.Progress()):
    return generate("lego", a[:16], a[16:], progress)


@friendly
def gen_cover(*a, progress=gr.Progress()):
    return generate("cover", a[:16], a[16:], progress)


# ----------------------------------------------------------------------------------------- mix
def pan(y, p):
    """Mono-sum equal-power pan (-1..1); 0 keeps the stereo file as it is."""
    if p == 0:
        return y
    m = y.mean(1, keepdims=True)
    a = (p + 1) * np.pi / 4
    return np.concatenate([m * np.cos(a), m * np.sin(a)], 1)


@friendly
def mix(a_json, b_json, m_voc, m_drm, m_bas, m_oth, m_a, m_b, p_a, p_b, master):
    if not a_json:
        raise gr.Error("Оберіть гітару A (спершу згенеруйте щось).")
    ma = json.load(open(a_json, encoding="utf-8"))
    root = ma["root"]  # the stems of the song THIS generation was made for
    layers = [load_audio(os.path.join(root, "orig", f"{k}.wav"), ma["start"], ma["end"]) * g
              for k, g in zip(STEMS, [m_voc, m_drm, m_bas, m_oth]) if g > 0]
    for j, g, p in [(a_json, m_a, p_a), (b_json, m_b, p_b)]:
        if j and g > 0:
            layers.append(pan(load_audio(json.load(open(j, encoding="utf-8"))["wav"]), p) * g)
    n = min(len(x) for x in layers)
    out = soft_limit(sum(x[:n] for x in layers) * master)
    path = os.path.join(root, f"{ma['name']}-{time.strftime('%Y%m%d-%H%M%S')}.wav")
    sf.write(path, out, SR)
    return playable(path), f"Збережено в кореневу папку: {os.path.basename(path)} ({n / SR:.0f} с, пік {np.abs(out).max():.2f})"


# ----------------------------------------------------------------------------------------- UI
def preset_choices():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "prompts", "*caption*.txt"))):
        for i, line in enumerate(open(p, encoding="utf-8")):
            if line.strip():
                out.append((f"{os.path.basename(p)}:{i} {line.strip()[:60]}", line.strip()))
    return out


def restore():
    """Re-open the last song without redoing the demux."""
    if os.path.exists(STATE_FILE):
        base = json.load(open(STATE_FILE, encoding="utf-8")).get("base")
        if base and os.path.exists(os.path.join(os.path.dirname(base), "orig", "full.wav")):
            root = os.path.dirname(os.path.abspath(base))
            state.update(base=base, root=root, name=os.path.splitext(os.path.basename(base))[0],
                         duration=sf.info(os.path.join(root, "orig", "full.wav")).duration)


def build():
    restore()
    h0 = history()
    with gr.Blocks(title="Реконструкція музики — тюнер") as demo:
        gr.Markdown("**Як користуватись:** 1) оберіть пісню — вона сама розділиться на стеми; 2) за бажанням оберіть "
                    "зразок звучання; 3) налаштуйте і натисніть «Згенерувати» у вкладці Lego або Cover; "
                    "4) у блоці «Зведення» крутіть повзунки та слухайте — без нової генерації.")
        gr.Markdown("### 1. Пісня")
        with gr.Row():
            base_path = gr.Textbox(state["base"] or "", label="Файл пісні", scale=4,
                                   info="Повний шлях до треку або кнопка «Огляд…». Папка цього файлу стає кореневою "
                                        "робочою папкою.")
            base_browse = gr.Button("Огляд…", scale=1)
            base_load = gr.Button("Завантажити пісню", variant="primary", scale=1)
        with gr.Row():
            root_box = gr.Textbox(state["root"] or "—", label="Коренева папка", interactive=False, scale=4,
                                  info="Тут лежать: готові результати (<назва>-<дата-час>.wav), підпапки orig, "
                                       "reference і gen.")
            open_btn = gr.Button("Відкрити папку", scale=1)
        song_info = gr.Markdown(song_status())

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 2. Що чує модель (контекст)")
                start = gr.Slider(0, int(state["duration"]), min(40, int(state["duration"])), step=1,
                                  label="Початок фрагмента, с",
                                  info="Яку частину пісні обробляти. Коротший фрагмент = швидша генерація "
                                       "(60 с ≈ 7 с).")
                end = gr.Slider(10, int(state["duration"]), min(100, int(state["duration"])), step=1,
                                label="Кінець фрагмента, с")
                g_voc = gr.Slider(0, 1.5, 1.0, step=0.05, label="Вокал у контексті",
                                  info="Наскільки голосно модель «чує» вокал. 0 = не чує зовсім.")
                g_drm = gr.Slider(0, 1.5, 1.0, step=0.05, label="Барабани у контексті",
                                  info="Допомагає моделі триматися ритму пісні.")
                g_bas = gr.Slider(0, 1.5, 1.0, step=0.05, label="Бас у контексті",
                                  info="Допомагає моделі потрапляти в акорди пісні.")
                g_oth = gr.Slider(0, 1.5, 0.0, step=0.05, label="Електроніка оригіналу у контексті",
                                  info="0 = модель не чує оригінальні синтезатори й замінює їх своєю партією; "
                                       "1 = дописує поверх них.")
            with gr.Column():
                gr.Markdown("### 3. Зразок звучання (необов'язково)")
                use_ref = gr.Checkbox(True, label="Використати зразок",
                                      info="Модель наслідує тембр і стиль зразка. Без нього стиль випадковий.")
                ref_name = gr.Dropdown(reference_names(), value=(reference_names() or [None])[0], label="Зразок",
                                       info="Список підготовлених зразків цієї пісні.")
                ref_stems = gr.CheckboxGroup(REF_STEM_CHOICES, value=["other"], label="Що брати зі зразка",
                                             info="Без вокалу — щоб слова зразка не потрапили у результат. Лише "
                                                  "«інше» — брати тембр гітар/клавіш, а не все аранжування.")
                ref_start = gr.Slider(0, 300, 0, step=1, label="Вікно зразка починається з, с",
                                      info="Модель завжди бере рівно 30 с зразка. Оберіть місце, де грає потрібна "
                                           "гітара.")
                with gr.Row():
                    ref_path = gr.Textbox(label="Новий зразок", scale=3, info="Шлях до файлу або «Огляд…».")
                    ref_browse = gr.Button("Огляд…", scale=1)
                ref_load = gr.Button("Завантажити зразок")
                ref_status = gr.Markdown("")
                ref_prev = gr.Audio(label="Зразок, який справді отримала модель", interactive=False)
            with gr.Column():
                gr.Markdown("### Спільні налаштування генерації")
                model = gr.Dropdown(MODELS, value=MODELS[0], label="Модель",
                                    info="base — потрібна для Lego. sft і turbo — для Cover (turbo швидша, 8 кроків).")
                caption = gr.Textbox(lines=4, label="Опис звуку (англійською)",
                                     value=(preset_choices() or [("", "")])[0][1],
                                     info="Текстовий опис потрібного звучання. Впливає слабо: головне — зразок і сід.")
                presets = gr.Dropdown(preset_choices(), label="Готові описи", info="Підставити опис з prompts/.")
                steps = gr.Slider(8, 100, 50, step=1, label="Кроки",
                                  info="Більше = детальніше, але повільніше. base/sft: ~50, turbo: 8.")
                guidance = gr.Slider(1, 15, 7, step=0.5, label="Guidance",
                                     info="Наскільки суворо модель слідує опису. Вище = слухняніше, але жорсткіший звук.")
                seed = gr.Number(0, precision=0, label="Сід",
                                 info="Число, що задає випадковість. Той самий сід і налаштування = той самий результат.")
                count = gr.Slider(1, 6, 1, step=1, label="Скільки варіантів",
                                  info="Зробити поспіль кілька варіантів: сід, сід+1, сід+2…")
        common = [model, caption, steps, guidance, seed, count, start, end, g_voc, g_drm, g_bas, g_oth,
                  use_ref, ref_name, ref_stems, ref_start]
        with gr.Tabs():
            with gr.Tab("Lego — дописати нову партію"):
                gr.Markdown("Модель дописує **нову доріжку** (напр. гітару) до того, що вона чує в контексті. "
                            "Повертається лише ця доріжка, барабани/бас/вокал додаються у зведенні.")
                track = gr.Dropdown(TRACKS, value="guitar", label="Яку партію дописати",
                                    info="Наприклад guitar — гітара.")
                rep_start = gr.Slider(0, 300, 0, step=1, label="Генерувати від, с",
                                      info="Нову партію лише на відрізку фрагмента. 0 і −1 = на весь фрагмент.")
                rep_end = gr.Slider(-1, 300, -1, step=1, label="Генерувати до, с (−1 = до кінця)")
                lego_btn = gr.Button("Згенерувати (Lego)", variant="primary")
            with gr.Tab("Cover — переграти те, що чує модель"):
                gr.Markdown("Модель **переграє** весь контекст у новому звучанні. Увага: велика сила джерела дає "
                            "майже копію оригіналу.")
                noise = gr.Slider(0, 1, 0.5, step=0.05, label="Шум джерела",
                                  info="0 = модель губить тональність і ритм; 0.4–0.6 = тримає їх.")
                strength = gr.Slider(0, 1, 0.7, step=0.05, label="Сила джерела",
                                     info="1.0 = результат майже повторює вхід; менше = модель вільніша.")
                cover_btn = gr.Button("Згенерувати (Cover)", variant="primary")
        status = gr.Markdown("")
        gen_audio = gr.Audio(label="Згенерована доріжка (окремо, без барабанів і вокалу)", interactive=False)

        gr.Markdown("### 4. Зведення (миттєво, без нової генерації)")
        with gr.Row():
            ga = gr.Dropdown(h0, value=h0[0][1] if h0 else None, label="Гітара A",
                             info="Одна з згенерованих доріжок (нові зверху).")
            gb = gr.Dropdown(h0, value=h0[1][1] if len(h0) > 1 else None, label="Гітара B (необов'язково)",
                             info="Друга доріжка: разом з A дає дві різні гітари.")
        with gr.Row():
            m_voc = gr.Slider(0, 2, 1.0, step=0.05, label="Вокал оригіналу", info="Чистий вокал із оригіналу.")
            m_drm = gr.Slider(0, 2, 1.0, step=0.05, label="Барабани оригіналу")
            m_bas = gr.Slider(0, 2, 1.0, step=0.05, label="Бас оригіналу")
            m_oth = gr.Slider(0, 2, 0.0, step=0.05, label="Електроніка оригіналу",
                              info="0 = прибрана (замінена гітарами).")
        with gr.Row():
            m_a = gr.Slider(0, 2, 1.0, step=0.05, label="Рівень гітари A")
            p_a = gr.Slider(-1, 1, 0.0, step=0.1, label="Панорама A", info="−1 ліворуч, 1 праворуч, 0 = як згенеровано.")
            m_b = gr.Slider(0, 2, 0.0, step=0.05, label="Рівень гітари B", info="0 = B вимкнено.")
            p_b = gr.Slider(-1, 1, 0.0, step=0.1, label="Панорама B")
            master = gr.Slider(0.2, 2, 1.0, step=0.05, label="Загальна гучність")
        mix_btn = gr.Button("Змішати", variant="primary")
        mix_audio = gr.Audio(label="Результат", interactive=False)
        mix_status = gr.Markdown("")

        mix_in = [ga, gb, m_voc, m_drm, m_bas, m_oth, m_a, m_b, p_a, p_b, master]
        song_out = [root_box, song_info, start, end, ref_name, ga, gb]
        gen_out = [gen_audio, ga, gb, ref_prev, status]
        base_browse.click(lambda: browse("Оберіть пісню"), None, base_path)
        base_load.click(load_song, base_path, song_out)
        open_btn.click(open_root)
        ref_browse.click(lambda: browse("Оберіть зразок"), None, ref_path)
        ref_load.click(load_reference, ref_path, [ref_name, ref_status])
        presets.change(lambda v: v, presets, caption)
        lego_btn.click(gen_lego, common + [track, rep_start, rep_end], gen_out).then(mix, mix_in, [mix_audio, mix_status])
        cover_btn.click(gen_cover, common + [noise, strength], gen_out).then(mix, mix_in, [mix_audio, mix_status])
        mix_btn.click(mix, mix_in, [mix_audio, mix_status])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    build().queue().launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.no_browser, show_error=True)
