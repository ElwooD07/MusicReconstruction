"""Experiment: ACE-Step 1.5 'cover' of the instrumental stem, mixed with the original vocals.

Run with the ACE-Step venv, via the launcher (paths come from scripts/config.ps1):
    run.bat acestep-cover <song_folder> [--strengths 0.5,0.7,1.0]
which executes:
    python tools/acestep_cover.py <song_folder> [out_dir] [--strengths 0.5,0.7,1.0]
Models are downloaded to ACESTEP_CHECKPOINTS_DIR on first use.
"""
import argparse
import os
import shutil
import sys
import time

import soundfile as sf
import torch
import torchaudio

ACESTEP_DIR = os.environ.get("ACESTEP_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ace-step")
ACESTEP_DIR = os.path.abspath(ACESTEP_DIR)
sys.path.insert(0, ACESTEP_DIR)
os.environ.setdefault("ACESTEP_PROJECT_ROOT", ACESTEP_DIR)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.model_downloader import ensure_dit_model, ensure_main_model, get_checkpoints_dir  # noqa: E402

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "default.txt")


def read_prompt(text):
    """Prompt given on the command line, otherwise the contents of prompts/default.txt (none in code)."""
    if text and text.strip():
        return text.strip()
    with open(PROMPT_FILE, encoding="utf-8") as f:
        return f.read().strip()


EXCERPT = (40, 100)  # seconds, same 60 s excerpt as the early experiments
SEED = 0


def cut(path, out_path, full=False):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:  # mono mp3 -> stereo like the demucs stems
        data = data[:, None].repeat(2, axis=1)
    sf.write(out_path, data if full else data[EXCERPT[0] * sr:EXCERPT[1] * sr], sr)
    return sr


def waveform_corr(a_path, b_path, max_lag_s=0.2):
    """Peak normalized cross-correlation of two files (mono, 16 kHz, within +-max_lag_s)."""
    import numpy as np

    def load16(p):
        x, sr_ = sf.read(p, dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        return torchaudio.functional.resample(torch.from_numpy(x), sr_, 16000).numpy()

    a, b = load16(a_path), load16(b_path)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lag = int(max_lag_s * 16000)
    cc = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)))
    cc = np.concatenate([cc[-lag:], cc[:lag + 1]]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.abs(cc).max())


def mix(rock_path, vocals_path, out_path):
    rock, r_sr = sf.read(rock_path, dtype="float32")
    vocals, v_sr = sf.read(vocals_path, dtype="float32")
    rock = torch.from_numpy(rock.T if rock.ndim > 1 else rock[None])
    vocals = torch.from_numpy(vocals.T if vocals.ndim > 1 else vocals[None])
    vocals = torchaudio.functional.resample(vocals, v_sr, r_sr)
    if vocals.shape[0] != rock.shape[0]:
        vocals, rock = vocals.mean(0, keepdim=True), rock.mean(0, keepdim=True)
    n = min(vocals.shape[1], rock.shape[1])
    out = vocals[:, :n] + rock[:, :n]
    peak = out.abs().max()
    if peak > 1.0:
        out = out / peak
    sf.write(out_path, out.T.numpy(), r_sr)
    return rock.shape[1] / r_sr, vocals.shape[1] / r_sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("out_dir", nargs="?", default="temp/ace")
    ap.add_argument("--strengths", default="0.5,0.7,1.0")
    ap.add_argument("--full", action="store_true", help="whole song instead of the 60 s excerpt")
    ap.add_argument("--caption", default=None,
                    help="style prompt text (default: contents of prompts/default.txt)")
    ap.add_argument("--bpm", type=int, default=None, help="tempo of the source, passed to the model")
    ap.add_argument("--tag", default="", help="suffix for output names, to keep variants apart")
    ap.add_argument("--seeds", default=str(SEED), help="comma-separated seeds, e.g. 0,1,2")
    ap.add_argument("--source", choices=["stems", "original", "vocals"], default="stems",
                    help="audio given to the model: instrumental stem (default), the original song with "
                         "vocals, or the vocals stem only (for --task complete/lego)")
    ap.add_argument("--src-file", default=None,
                    help="use this audio file (e.g. a single instrument stem) as the source instead")
    ap.add_argument("--task", choices=["cover", "complete", "lego"], default="cover",
                    help="complete/lego need a base or SFT model (--model acestep-v15-sft)")
    ap.add_argument("--instruction", default=None, help="instruction for complete/lego")
    ap.add_argument("--model", default="acestep-v15-turbo",
                    help="DiT checkpoint, e.g. acestep-v15-turbo, acestep-v15-sft, acestep-v15-base")
    ap.add_argument("--steps", type=int, default=None, help="inference steps (turbo 8, sft/base 50)")
    ap.add_argument("--guidance", type=float, default=None,
                    help="guidance_scale: how strictly the model follows the caption (default 7.0; not used by turbo)")
    ap.add_argument("--stems-dir", default=None,
                    help="folder with vocals.wav and no_vocals.wav (default <song>/separated/htdemucs/<name>)")
    ap.add_argument("--noise", default="0",
                    help="comma-separated cover_noise_strength values: 0 = ignore the source audio "
                         "(style from the caption only), 1 = stay closest to the source audio")
    ap.add_argument("--reference", default=None,
                    help="audio file whose sound/timbre guides the generation (ACE-Step reference_audio)")
    ap.add_argument("--captions-file", default=None,
                    help="one caption per line: every caption is run for every seed (names get c<index>)")
    ap.add_argument("--sing", action="store_true",
                    help="let the model render the vocals too (single track, nothing is mixed back in)")
    ap.add_argument("--lyrics-file", default=None, help="UTF-8 lyrics for --sing (default: none)")
    args = ap.parse_args()
    lyrics = "[Instrumental]"
    if args.sing:
        lyrics = open(args.lyrics_file, encoding="utf-8").read() if args.lyrics_file else ""
    caption = read_prompt(args.caption)
    tag = f"{args.tag}_" if args.tag else ""
    out_dir = os.path.abspath(args.out_dir) + ("_full" if args.full else "")
    os.makedirs(out_dir, exist_ok=True)

    folder = os.path.abspath(args.folder)
    song = [f for f in os.listdir(folder) if f.lower().endswith(".mp3")][0][:-4]
    stems = os.path.abspath(args.stems_dir) if args.stems_dir else os.path.join(
        folder, "separated", "htdemucs", song)
    src = os.path.join(out_dir, "src_excerpt_no_vocals.wav")
    voc = os.path.join(out_dir, "src_excerpt_vocals.wav")
    cut(os.path.join(stems, "no_vocals.wav"), src, args.full)
    cut(os.path.join(stems, "vocals.wav"), voc, args.full)
    if args.source == "original":  # cover the full mix; stems are still used for the optional vocal mix
        src = os.path.join(out_dir, "src_excerpt_original.wav")
        cut(os.path.join(folder, song + ".mp3"), src, args.full)
    elif args.source == "vocals":
        src = voc
    if args.src_file:
        src = os.path.join(out_dir, "src_excerpt_custom.wav")
        cut(os.path.abspath(args.src_file), src, args.full)

    ckpt = get_checkpoints_dir()
    print("checkpoints:", ckpt, flush=True)
    ok, msg = ensure_main_model(checkpoints_dir=ckpt)
    print("models:", ok, msg, flush=True)
    if ok and args.model != "acestep-v15-turbo":
        ok, msg = ensure_dit_model(args.model, checkpoints_dir=ckpt)
        print(f"model {args.model}:", ok, msg, flush=True)
    if not ok:
        sys.exit(1)

    dit = AceStepHandler()
    status, ok = dit.initialize_service(project_root=ACESTEP_DIR, config_path=args.model,
                                        device="cuda")
    print("dit init:", ok, status, flush=True)
    if not ok:
        sys.exit(1)
    llm = LLMHandler()  # not initialized: the LM is skipped for cover tasks

    captions = [l.strip() for l in open(args.captions_file, encoding="utf-8") if l.strip()]         if args.captions_file else [caption]
    seeds = [int(x) for x in args.seeds.split(",")]
    noises = [float(x) for x in args.noise.split(",")]
    for ci, caption, s, ns, seed in [(ci, cap, s, ns, seed) for ci, cap in enumerate(captions)
                                     for s in [float(x) for x in args.strengths.split(",")]
                                     for ns in noises for seed in seeds]:
        if True:
            t0 = time.time()
            name = f"{tag}{'c%d_' % ci if len(captions) > 1 else ''}{s}" + (f"_n{ns}" if len(noises) > 1 or ns else "") + \
                   (f"_s{seed}" if len(seeds) > 1 else "")
            save_dir = os.path.join(out_dir, f"cover_{name}")
            kw = dict(task_type=args.task, src_audio=src, caption=caption, lyrics=lyrics,
                      instrumental=(args.task == "cover" and not args.sing),
                      duration=-1, seed=seed, thinking=False, bpm=args.bpm)
            if args.task == "cover":
                kw.update(audio_cover_strength=s, cover_noise_strength=ns)
            elif args.instruction:
                kw.update(instruction=args.instruction)
            if args.reference:
                kw.update(reference_audio=os.path.abspath(args.reference))
            if args.steps:
                kw.update(inference_steps=args.steps)
            if args.guidance:
                kw.update(guidance_scale=args.guidance)
            params = GenerationParams(**kw)
            config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False,
                                      seeds=[seed])
            result = generate_music(dit, llm, params, config, save_dir=save_dir)
            if not result.success:
                print(f"strength {s} seed {seed}: FAILED: {getattr(result, 'error', result)}", flush=True)
                continue
            rock = result.audios[0]["path"]
            mixed = os.path.join(out_dir, f"mix_{args.task}_{name}.wav")
            extra = ""
            if args.sing or args.task != "cover":  # the output may already contain the vocals
                shutil.copyfile(rock, mixed)
                data, sr_ = sf.read(mixed)
                r_len = v_len = len(data) / sr_
                # does the output contain the original vocal waveform? (~0 = no, high = yes)
                extra = f"  vocal_waveform_corr={waveform_corr(rock, voc):.2f}"
                if args.task != "cover":  # also offer output + original vocals
                    mix(rock, voc, os.path.join(out_dir, f"mixv_{args.task}_{name}.wav"))
            else:
                r_len, v_len = mix(rock, voc, mixed)
            print(f"strength {s} seed {seed}: {round(time.time() - t0, 1)}s  rock={r_len:.1f}s "
                  f"vocals={v_len:.1f}s{extra} -> {mixed}", flush=True)


if __name__ == "__main__":
    main()
