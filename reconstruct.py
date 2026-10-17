import os
import sys
import subprocess

import torch
import numpy as np
from scipy.io import wavfile
from audiocraft.models import MusicGen
import torchaudio
from audiocraft.data.audio_utils import normalize_audio

# ==================== НАЛАШТУВАННЯ META AI ====================
MODEL_NAME = 'facebook/musicgen-melody-large'
CFG_SCALE = 3.8
DEFAULT_PROMPT = "Modern hard rock track, heavy overdriven electric guitars, punchy modern drums, thick bassline, professional stadium rock mix, keeping original harmony"
# ==============================================================

def main():
    print("\n--- 🚀 Старт оригінального фреймворку Meta AudioCraft на RTX 4080 ---")
    
    if len(sys.argv) < 2:
        print("❌ Помилка: Не вказано папку з файлом!")
        sys.exit(1)

    target_folder = os.path.abspath(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[2].strip() else DEFAULT_PROMPT

    if not os.path.exists(target_folder):
        print("❌ Помилка: Папку не знайдено!")
        sys.exit(1)

    mp3_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.mp3')]
    if not mp3_files:
        print("❌ Помилка: У папці немає .mp3 файлів!")
        sys.exit(1)
    
    mp3_file = os.path.join(target_folder, mp3_files[0])
    song_name = os.path.splitext(mp3_files[0])[0]

    # 1. Розділення Demucs (файли вже готові, етап пройде миттєво)
    print("1/3. Перевірка файлів вокалу...")
    base_separated = os.path.join(target_folder, "separated", "htdemucs", song_name)
    VOCALS_PATH = os.path.join(base_separated, "vocals.wav")
    NO_VOCALS_PATH = os.path.join(base_separated, "no_vocals.wav")

    # 2. Завантаження моделі безпосередньо в CUDA пам'ять відеокарти
    print(f"2/3. Завантаження моделі Meta {MODEL_NAME} прямо у VRAM...")
    model = MusicGen.get_pretrained(MODEL_NAME, device='cuda')

    # Зчитуємо старий фон через scipy
    sr, melody_data = wavfile.read(NO_VOCALS_PATH)
    
    # Конвертуємо в нативний тензор PyTorch для відеокарти
    melody_tensor = torch.from_numpy(melody_data).float()
    if melody_data.dtype == np.int16:
        melody_tensor = melody_tensor / 32768.0
    elif melody_data.dtype == np.int32:
        melody_tensor = melody_tensor / 2147483648.0
    
    # Вирівнюємо розмірність під вимоги Meta AI [channels, samples]
    if len(melody_tensor.shape) == 1:
        melody_tensor = melody_tensor.unsqueeze(0)
    elif len(melody_tensor.shape) == 2:
        # scipy.io.wavfile повертає [samples, channels]
        melody_tensor = melody_tensor.T.contiguous()

    duration = int(melody_tensor.shape[1] / sr)
    
    # Передаємо параметри генерації в модель
    model.set_generation_params(duration=duration, cfg_coef=CFG_SCALE)

    print(f"🔥 АКТИВАЦІЯ CUDA: Повний прорахунок треку ({duration} сек.) на ядрах RTX 4080...")
    with torch.no_grad():
        # Рідний метод хрома-структур Meta AI. Працює суто на відеокарті!
        outputs = model.generate_with_chroma([prompt], melody_tensor[None].to('cuda'), sr)

    # Нормалізуємо гучність без ffmpeg і зберігаємо рок-фон окремо (проміжний результат)
    new_rock_bg = outputs.cpu().squeeze(0)
    new_rock_bg = normalize_audio(new_rock_bg, strategy="loudness", loudness_compressor=True,
                                  sample_rate=model.sample_rate)
    rock_path = os.path.join(target_folder, f"{song_name}_rock_only.wav")
    wavfile.write(rock_path, rate=model.sample_rate,
                  data=(new_rock_bg.mean(dim=0).clamp(-1, 1).numpy() * 32767.0).astype(np.int16))

    # 3. Фінальне зведення
    print("3/3. Зведення нового року з оригінальним вокалом...")
    v_sr, vocals_data = wavfile.read(VOCALS_PATH)

    vocals_tensor = torch.from_numpy(vocals_data).float()
    if vocals_data.dtype == np.int16:
        vocals_tensor = vocals_tensor / 32768.0
    elif vocals_data.dtype == np.int32:
        vocals_tensor = vocals_tensor / 2147483648.0
    if len(vocals_tensor.shape) > 1:
        vocals_tensor = vocals_tensor.mean(dim=1)

    # MusicGen видає 32 кГц, вокал має іншу частоту - приводимо рок до частоти вокалу
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
    print("🎉 УСПІХ! ОФІЦІЙНИЙ КОД META AI ПОВНІСТЮ ЗГЕНЕРУВАВ ТРЕК!")
    print(f"👉 Результат збережено тут:\nℹ️  {os.path.abspath(output_path)}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
