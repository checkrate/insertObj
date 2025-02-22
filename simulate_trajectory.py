import numpy as np
from scipy.signal import medfilt
from typing import List, Tuple

def log_execution(func):
    import time, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TrajectorySimulator] {func.__name__} выполнена за {end - start:.4f} секунд")
        return result
    return wrapper

class TrajectorySimulator:
    """
    Класс для симуляции траектории на основе ключевых кадров,
    считываемых из файла.
    """
    def __init__(self, input_file: str, n_frames: int, noise_amplitude: float = 2.0, acceleration_variation: float = 0.15):
        """
        :param input_file: Путь к файлу с ключевыми кадрами.
        :param n_frames: Общее количество кадров.
        :param noise_amplitude: Амплитуда шума.
        :param acceleration_variation: Вариация ускорения.
        """
        self.input_file = input_file
        self.n_frames = n_frames
        self.noise_amplitude = noise_amplitude
        self.acceleration_variation = acceleration_variation
        self.key_frames = self._read_keyframes()

    def _read_keyframes(self) -> List[Tuple[int, float, float, float]]:
        """
        Считывает ключевые кадры из файла.
        Формат строки: frame_number x_coordinate y_coordinate [s_coordinate]
        """
        key_frames = []
        try:
            with open(self.input_file, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) >= 3:
                        frame_number = int(tokens[0])
                        x_coordinate = float(tokens[1])
                        y_coordinate = float(tokens[2])
                        s_coordinate = float(tokens[3]) if len(tokens) >= 4 else 1.0
                        key_frames.append((frame_number, x_coordinate, y_coordinate, s_coordinate))
        except FileNotFoundError:
            print(f"Warning: Файл {self.input_file} не найден. Будет использована траектория по умолчанию.")
        return key_frames

    @log_execution
    def simulate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Симулирует траекторию по ключевым кадрам.
        
        :return: Тройка numpy-массивов с координатами x, y и масштабом s.
        """
        n = self.n_frames
        positions_x = np.zeros(n)
        positions_y = np.zeros(n)
        positions_s = np.ones(n)
        
        if not self.key_frames:
            return positions_x, positions_y, positions_s
        
        # Сортировка ключевых кадров по номеру кадра
        self.key_frames.sort(key=lambda x: x[0])
        first_frame, first_x, first_y, first_s = self.key_frames[0]
        
        if first_frame > 0:
            positions_x[:first_frame] = first_x
            positions_y[:first_frame] = first_y
            positions_s[:first_frame] = first_s
        
        for i in range(len(self.key_frames) - 1):
            frame_start, x_start, y_start, s_start = self.key_frames[i]
            frame_end, x_end, y_end, s_end = self.key_frames[i + 1]
            frames_range = frame_end - frame_start
            if frames_range <= 0:
                continue
            t = np.linspace(0, 1, frames_range + 1)
            linear_x = x_start + (x_end - x_start) * t
            linear_y = y_start + (y_end - y_start) * t
            linear_s = s_start + (s_end - s_start) * t
            noise_x = self.noise_amplitude * np.random.uniform(-1, 1, frames_range + 1)
            noise_y = self.noise_amplitude * np.random.uniform(-1, 1, frames_range + 1)
            acceleration = 1 + self.acceleration_variation * np.sin(2 * np.pi * t * np.random.uniform(0.8, 1.2))
            positions_x[frame_start:frame_end + 1] = linear_x + noise_x * acceleration
            positions_y[frame_start:frame_end + 1] = linear_y + noise_y * acceleration
            positions_s[frame_start:frame_end + 1] = linear_s
        
        last_frame, last_x, last_y, last_s = self.key_frames[-1]
        if last_frame < n - 1:
            positions_x[last_frame:] = last_x
            positions_y[last_frame:] = last_y
            positions_s[last_frame:] = last_s
        
        if len(self.key_frames) == 1:
            positions_x[:] = first_x
            positions_y[:] = first_y
            positions_s[:] = first_s
        
        # медианная фильтрации для сглаживания
        positions_x = medfilt(positions_x, kernel_size=5)
        positions_y = medfilt(positions_y, kernel_size=5)
        positions_s = medfilt(positions_s, kernel_size=5)
        
        return positions_x, positions_y, positions_s

if __name__ == "__main__":
    # Пример использования:
    simulator = TrajectorySimulator("input.txt", n_frames=100)
    pos_x, pos_y, pos_s = simulator.simulate()
    print("Симуляция завершена.")
