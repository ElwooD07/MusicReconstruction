import os
import sys
import subprocess
import types

# ХАК ДЛЯ ОБХОДУ БЛОКУВАННЯ XFORMERS НА WINDOWS:
# Створюємо фейковий модуль у пам'яті, щоб застарілий код Meta не падав при імпорті
if 'xformers' not in sys.modules:
    fake_xformers = types.ModuleType('xformers')
    fake_xformers.ops = types.ModuleType('ops')
    # Заглушка для методу memory_efficient_attention, якщо модель спробує його викликати
    fake_xformers.ops.memory_efficient_attention = None
    sys.modules['xformers'] = fake_xformers
    sys.modules['xformers.ops'] = fake_xformers.ops

# Тепер імпорти пройдуть без жодних помилок 'ModuleNotFoundError'
import torch
import numpy as np
from scipy.io import wavfile
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

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
    elif len(melody_tensor.shape) == 2 and melody_tensor.shape[1] > melody_tensor.shape[0]:
        melody_tensor = melody_tensor.T

    duration = int(melody_tensor.shape[1] / sr)
    
    # Передаємо параметри генерації в модель
    model.set_generation_params(duration=duration, cfg_scale=CFG_SCALE)

    print(f"🔥 АКТИВАЦІЯ CUDA: Повний прорахунок треку ({duration} сек.) на ядрах RTX 4080...")
    with torch.no_grad():
        # Рідний метод хрома-структур Meta AI. Працює суто на відеокарті!
        outputs = model.generate_with_chroma([prompt], melody_tensor[None].to('cuda'), sr)

    # Переносимо результат і зберігаємо тимчасовий рок-фон
    new_rock_bg = outputs.cpu().squeeze(0)
    audio_write("temp_meta_rock", new_rock_bg, model.sample_rate, strategy="loudness", loudness_compressor=True)

    # 3. Фінальне зведення
    print("3/3. Зведення нового року з оригінальним вокалом...")
    v_sr, vocals_data = wavfile.read(VOCALS_PATH)
    r_sr, rock_data = wavfile.read("temp_meta_rock.wav")

    vocals_tensor = torch.from_numpy(vocals_data).float()
    if len(vocals_tensor.shape) > 1:
        vocals_tensor = vocals_tensor.mean(dim=1)

    rock_tensor = torch.from_numpy(rock_data).float()
    if len(rock_tensor.shape) > 1:
        rock_tensor = rock_tensor.mean(dim=1)

    min_len = min(vocals_tensor.shape[0], rock_tensor.shape[0])
    final_mix = vocals_tensor[:min_len] + rock_tensor[:min_len]

    max_val = torch.max(torch.abs(final_mix))
    if max_val > 1.0:
        final_mix = final_mix / max_val

    output_path = os.path.join(target_folder, f"{song_name}_RECONSTRUCTED.wav")
    wavfile.write(output_path, rate=r_sr, data=(final_mix.numpy() * 32767.0).astype(np.int16))

    if os.path.exists("temp_meta_rock.wav"):
        os.remove("temp_meta_rock.wav")

    print("\n=======================================================")
    print("🎉 УСПІХ! ОФІЦІЙНИЙ КОД META AI ПОВНІСТЮ ЗГЕНЕРУВАВ ТРЕК!")
    print(f"👉 Результат збережено тут:\nℹ️  {os.path.abspath(output_path)}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
