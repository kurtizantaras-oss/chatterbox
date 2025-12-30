import numpy as np
import torch
from scipy.signal import resample
import random

modelvad, utilsvad = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              onnx=True,             # Key: Use the ONNX version of the model
                                              force_onnx_cpu=True,   # Optional: Force ONNX Runtime to use CPU
                                              force_reload=False)

def trim_audio_with_pauses(wav_numpy, add_sylense=True, sr=16000):
    """
    Обнаруживает речь в аудиофайле с помощью Silero VAD, обрезает тишину
    и добавляет случайные паузы в начале и конце.

    Входные и выходные данные - NumPy-массивы.
    Использует Silero VAD с ONNX Runtime, форсируя CPU.

    Args:
        wav_numpy (np.ndarray): Входной аудиомассив в формате NumPy.
        sr (int): Частота дискретизации (должна быть 8000 или 16000).

    Returns:
        np.ndarray: Обрезанный аудиомассив с паузами или исходный, если речь не найдена.
    """

    # 1. Проверка типа входного аргумента
    if not isinstance(wav_numpy, np.ndarray):
        raise ValueError("Входные данные должны быть NumPy-массивом.")

    # 2. Извлечение необходимой VAD-утилиты
    # Извлекаем функцию get_speech_timestamps из глобального кортежа utilsvad
    (get_speech_timestamps, _, _, _, _) = utilsvad

    # 3. Подготовка данных для VAD: Преобразование NumPy в PyTorch Tensor
    # Silero VAD ожидает входные данные в формате PyTorch Tensor.
    wav_tensor = torch.from_numpy(wav_numpy)

    # 4. Обнаружение временных меток речи
    print("Try to detect speech...")
    speech_timestamps = get_speech_timestamps(wav_tensor,
                                              modelvad,
                                              sampling_rate=sr,
                                              # Порог активности голоса
                                              threshold=0.6,
                                              # Минимальная длительность речи, чтобы считаться активной
                                              min_speech_duration_ms=250,
                                              # Минимальная длительность тишины между сегментами речи
                                              min_silence_duration_ms=50,
                                              )

    # 5. Проверка результата VAD
    if not speech_timestamps:
        print("Speech is not in audio. Return original.")
        # ⚠️ Возвращается пустой массив нулей, а не исходный файл.
        return wav_numpy

    # 6. Определение границ общего сегмента речи
    # Начало - это начало первого обнаруженного сегмента.
    start_sample = speech_timestamps[0]['start']
    # Конец - это конец последнего обнаруженного сегмента.
    end_sample = speech_timestamps[-1]['end']

    # 7. Обрезка основного сегмента речи
    # Извлечение всех сэмплов от начала первого до конца последнего сегмента речи.
    speech_segment_numpy = wav_numpy[start_sample:end_sample]
    
    if add_sylense:
        # 8. Генерация случайной паузы
        # Выбираем случайную длительность паузы в секундах (от 0.2 до 0.5)
        random_pause_sec = random.uniform(0.2, 0.4)
        
        # 9. Преобразование длительности паузы в количество сэмплов
        pause_samples = int(random_pause_sec * sr)
        
        # 10. Создание массива тишины (нулей)
        # Массив сэмплов, представляющий тишину/паузу.
        silence_numpy = np.zeros(pause_samples, dtype=wav_numpy.dtype)
        # 11. Объединение сегментов аудио
        # ⚠️ Текущая реализация объединяет ТОЛЬКО речь и паузу в конце.
        # [Пауза в конце] + [Речь]
        final_audio_numpy = np.concatenate([silence_numpy, speech_segment_numpy])
    else:
        final_audio_numpy = speech_segment_numpy

    # 12. Возврат обработанного аудио
    return final_audio_numpy
    
def normalize_peak_numpy(data: np.ndarray, coefficient: float = 1.0) -> np.ndarray:
    """
    Нормализует аудио (NumPy) по максимальному пику.
    """
    # Исправлено: np.max ищет только положительный максимум.
    # Для аудио нужно np.abs(data).max(), чтобы учесть громкие отрицательные значения.
    max_value = np.max(np.abs(data))
    if max_value > 0:
        data = data / max_value
    return data * coefficient
    
def resample_wav(audio: np.ndarray, audio_sr: int, target_sr: int=16000) -> np.ndarray:
    """
    Изменяет частоту дискретизации аудио.
    """
    if audio_sr == target_sr:
        return audio
    num_original_samples = len(audio)
    num_target_samples = int(num_original_samples * (int(target_sr) / int(audio_sr)))
    return resample(audio, num_target_samples)
    
def normalize_peak_numpy(data: np.ndarray, coefficient: float = 1.0) -> np.ndarray:
    """
    Нормализует аудио (NumPy) по максимальному пику.
    """
    # Исправлено: np.max ищет только положительный максимум.
    # Для аудио нужно np.abs(data).max(), чтобы учесть громкие отрицательные значения.
    max_value = np.max(np.abs(data))
    if max_value > 0:
        data = data / max_value
    return data * coefficient