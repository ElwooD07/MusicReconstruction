import os
import sys
import subprocess

import torch
import numpy as np
from scipy.io import wavfile
from audiocraft.models import MusicGen
import torchaudio
from audiocraft.data.audio_utils import normalize_audio

# ==================== META AI SETTINGS ====================
MODEL_NAME = 'facebook/musicgen-melody-large'
CFG_SCALE = 3.8
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts', 'default.txt')
# ==============================================================

def load_prompt(arg):
    # Prompt: the command-line argument, otherwise the content of prompts/default.txt (no prompt text in the code)
    if arg and arg.strip():
        return arg.strip()
    if not os.path.exists(PROMPT_FILE):
        print(f"Error: no prompt given and no file {PROMPT_FILE}")
        sys.exit(1)
    with open(PROMPT_FILE, encoding='utf-8') as f:
        return f.read().strip()

def main():
    print("\n--- Starting the original Meta AudioCraft framework on the GPU ---")
    
    if len(sys.argv) < 2:
        print("Error: no folder with the song given!")
        sys.exit(1)

    target_folder = os.path.abspath(sys.argv[1])
    prompt = load_prompt(sys.argv[2] if len(sys.argv) >= 3 else None)

    if not os.path.exists(target_folder):
        print("Error: folder not found!")
        sys.exit(1)

    mp3_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.mp3')]
    if not mp3_files:
        print("Error: no .mp3 files in the folder!")
        sys.exit(1)
    
    mp3_file = os.path.join(target_folder, mp3_files[0])
    song_name = os.path.splitext(mp3_files[0])[0]

    # 1. Demucs separation (the files are already there, this step is instant)
    print("1/3. Checking the vocal files...")
    base_separated = os.path.join(target_folder, "separated", "htdemucs", song_name)
    VOCALS_PATH = os.path.join(base_separated, "vocals.wav")
    NO_VOCALS_PATH = os.path.join(base_separated, "no_vocals.wav")

    # 2. Load the model straight into the GPU memory (CUDA)
    print(f"2/3. Loading the Meta model {MODEL_NAME} straight into VRAM...")
    model = MusicGen.get_pretrained(MODEL_NAME, device='cuda')

    # Read the old backing with scipy
    sr, melody_data = wavfile.read(NO_VOCALS_PATH)
    
    # Convert to a native PyTorch tensor for the GPU
    melody_tensor = torch.from_numpy(melody_data).float()
    if melody_data.dtype == np.int16:
        melody_tensor = melody_tensor / 32768.0
    elif melody_data.dtype == np.int32:
        melody_tensor = melody_tensor / 2147483648.0
    
    # Shape it the way Meta AI expects: [channels, samples]
    if len(melody_tensor.shape) == 1:
        melody_tensor = melody_tensor.unsqueeze(0)
    elif len(melody_tensor.shape) == 2:
        # scipy.io.wavfile returns [samples, channels]
        melody_tensor = melody_tensor.T.contiguous()

    duration = int(melody_tensor.shape[1] / sr)
    
    # Pass the generation parameters to the model
    model.set_generation_params(duration=duration, cfg_coef=CFG_SCALE)

    print(f"CUDA: generating the whole track ({duration} s) on the GPU...")
    with torch.no_grad():
        # Meta AI's native chroma-conditioned method. Runs on the GPU only
        outputs = model.generate_with_chroma([prompt], melody_tensor[None].to('cuda'), sr)

    # Normalise the loudness without ffmpeg and save the rock backing separately (an intermediate result)
    new_rock_bg = outputs.cpu().squeeze(0)
    new_rock_bg = normalize_audio(new_rock_bg, strategy="loudness", loudness_compressor=True,
                                  sample_rate=model.sample_rate)
    rock_path = os.path.join(target_folder, f"{song_name}_rock_only.wav")
    wavfile.write(rock_path, rate=model.sample_rate,
                  data=(new_rock_bg.mean(dim=0).clamp(-1, 1).numpy() * 32767.0).astype(np.int16))

    # 3. Final mix
    print("3/3. Mixing the new rock with the original vocals...")
    v_sr, vocals_data = wavfile.read(VOCALS_PATH)

    vocals_tensor = torch.from_numpy(vocals_data).float()
    if vocals_data.dtype == np.int16:
        vocals_tensor = vocals_tensor / 32768.0
    elif vocals_data.dtype == np.int32:
        vocals_tensor = vocals_tensor / 2147483648.0
    if len(vocals_tensor.shape) > 1:
        vocals_tensor = vocals_tensor.mean(dim=1)

    # MusicGen outputs 32 kHz and the vocals have another rate: resample the rock to the vocal rate
    rock_tensor = new_rock_bg.mean(dim=0)
    rock_tensor = torchaudio.functional.resample(rock_tensor, model.sample_rate, v_sr)

    min_len = min(vocals_tensor.shape[0], rock_tensor.shape[0])
    final_mix = vocals_tensor[:min_len] + rock_tensor[:min_len]

    max_val = torch.max(torch.abs(final_mix))
    if max_val > 1.0:
        final_mix = final_mix / max_val

    output_path = os.path.join(target_folder, f"{song_name}_RECONSTRUCTED.wav")
    wavfile.write(output_path, rate=v_sr, data=(final_mix.numpy() * 32767.0).astype(np.int16))

    print("\n=======================================================")
    print("DONE: the track was generated by the official Meta AI code.")
    print(f"Result saved here:\n  {os.path.abspath(output_path)}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
