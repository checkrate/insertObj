import os
import os.path as osp
import cv2
import shutil
import tempfile
import subprocess
import numpy as np
from argparse import ArgumentParser
from typing import Tuple, List

def log_execution(func):
    import time, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[{func.__qualname__}] выполнена за {end - start:.4f} секунд")
        return result
    return wrapper

class VideoProcessor:
    """
    Класс для обработки видео: извлечение кадров, вычисление движения камеры,
    рендеринг видео и объединение с траекторией.
    """
    def __init__(self, video_path: str, save_dir: str = "save", res_dir: str = "res"):
        """
        :param video_path: Путь к исходному видео.
        :param save_dir: Директория для сохранения промежуточных файлов.
        :param res_dir: Директория для сохранения результата.
        """
        self.video_path = video_path
        self.save_dir = save_dir
        self.res_dir = res_dir
        self.raw_frames_path = osp.join(self.save_dir, "raw_frames")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.raw_frames_path, exist_ok=True)
        # Создаем детектор
        self.detector = cv2.AKAZE_create()

    @log_execution
    def parse_video(self) -> Tuple[int, Tuple[int, int]]:
        """
        Извлекает кадры из видео и сохраняет их в self.raw_frames_path.
        
        :return: Кортеж из числа кадров и размера кадра (высота, ширина).
        """
        vidcap = cv2.VideoCapture(self.video_path)
        success, frame = vidcap.read()
        frame_size = None
        count = 0
        while success:
            frame_filename = osp.join(self.raw_frames_path, f"frame{count}.png")
            if frame is not None:
                frame_size = frame.shape[:2]
                cv2.imwrite(frame_filename, frame)
            else:
                print(f"Warning: Frame {count} is пустой.")
            success, frame = vidcap.read()
            count += 1
        if count == 0:
            raise ValueError("Не удалось извлечь кадры из видео.")
        return count, frame_size

    @log_execution
    def render_video(self, output_video_path: str, frames: np.ndarray, frames_dir: str, fps: int) -> None:
        """
        Собирает видео из последовательности кадров.
        
        :param output_video_path: Путь для сохранения итогового видео.
        :param frames: Массив индексов кадров.
        :param frames_dir: Директория с кадрами.
        :param fps: Частота кадров итогового видео.
        """
        first_frame_path = osp.join(frames_dir, f"frame{frames[0]}.png")
        img = cv2.imread(first_frame_path)
        if img is None:
            raise ValueError(f"Первый кадр не найден: {first_frame_path}")
        frameSize = (img.shape[1], img.shape[0])
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frameSize)
        for i in frames:
            filename = osp.join(frames_dir, f"frame{i}.png")
            if osp.exists(filename):
                img = cv2.imread(filename)
                if img is not None:
                    frame = cv2.resize(img, frameSize)
                    out.write(frame)
                else:
                    print(f"Warning: Frame {i} пустой.")
            else:
                print(f"Warning: Frame {i} отсутствует.")
        out.release()

    @log_execution
    def compute_camera_motion(self, num_frames: int, threshold: float = 0.01) -> List[np.ndarray]:
        """
        Вычисляет движение камеры между последовательными кадрами.
        
        :param num_frames: Количество кадров.
        :param threshold: Пороговое значение для определения значимого смещения.
        :return: Список матриц движения (3x3).
        """
        motions = []
        prev_frame = None
        for i in range(num_frames):
            frame_path = osp.join(self.raw_frames_path, f"frame{i}.png")
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                motions.append(np.eye(3, dtype=np.float32))
                continue
            if prev_frame is None:
                prev_frame = frame
                motions.append(np.eye(3, dtype=np.float32))
                continue
            kp1, des1 = self.detector.detectAndCompute(prev_frame, None)
            kp2, des2 = self.detector.detectAndCompute(frame, None)
            if des1 is not None and des2 is not None:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                if matches:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if matrix is not None:
                        M_3x3 = np.eye(3, dtype=np.float32)
                        M_3x3[:2, :] = matrix
                        translation_change = np.linalg.norm(M_3x3[:2, 2])
                        if translation_change > threshold:
                            motions.append(M_3x3)
                        else:
                            motions.append(np.eye(3, dtype=np.float32))
                    else:
                        motions.append(np.eye(3, dtype=np.float32))
                else:
                    motions.append(np.eye(3, dtype=np.float32))
            else:
                motions.append(np.eye(3, dtype=np.float32))
            prev_frame = frame
        return motions

    @staticmethod
    def accumulate_motions(motions: List[np.ndarray]) -> List[np.ndarray]:
        """
        Накопление последовательных движений камеры.
        
        :param motions: Список матриц движения.
        :return: Список глобальных движений.
        """
        num_frames = len(motions)
        global_motions = [np.eye(3, dtype=np.float32) for _ in range(num_frames)]
        for i in range(1, num_frames):
            global_motions[i] = motions[i] @ global_motions[i - 1]
        return global_motions

    @staticmethod
    def project_positions_with_camera_motion(positions: Tuple[np.ndarray, np.ndarray],
                                             global_motions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Проецирует мировые координаты с учетом движения камеры.
        
        :param positions: Два numpy-массива с координатами x и y.
        :param global_motions: Список глобальных матриц движения камеры.
        :return: Два numpy-массива с откорректированными координатами.
        """
        n_frames = len(positions[0])
        adjusted_x = np.zeros(n_frames)
        adjusted_y = np.zeros(n_frames)
        for i in range(n_frames):
            M = global_motions[i]
            Xw = positions[0][i]
            Yw = positions[1][i]
            pt_world = np.array([Xw, Yw, 1.0], dtype=np.float32)
            pt_cam = M @ pt_world
            adjusted_x[i] = pt_cam[0]
            adjusted_y[i] = pt_cam[1]
        return adjusted_x, adjusted_y

    def _get_output_video_name(self) -> str:
        base = osp.splitext(osp.basename(self.video_path))[0]
        return f"{base}_synced.mp4"

    @log_execution
    def process_video(self, path_to_object: str, path_to_mask: str, fps: int, mask_method: str, insert_method: str) -> None:
        """
        Полный pipeline обработки видео: извлечение кадров, вычисление движения камеры,
        симуляция траектории, вставка объекта и рендеринг итогового видео.
        
        :param path_to_object: Путь к изображению объекта.
        :param path_to_mask: Путь к изображению маски.
        :param fps: FPS выходного видео.
        :param mask_method: Метод получения маски ("auto" или "grab").
        :param insert_method: Метод вставки объекта ("poisson" или "simple").
        """
        # Извлечение кадров
        num_frames, frame_size = self.parse_video()
        # Вычисление движения камеры
        motions = self.compute_camera_motion(num_frames)
        global_motions = self.accumulate_motions(motions)
        # Симуляция траектории 
        from simulate_trajectory import TrajectorySimulator
        simulator = TrajectorySimulator("input.txt", n_frames=num_frames)
        pos_x, pos_y, pos_s = simulator.simulate()
        adjusted_x, adjusted_y = self.project_positions_with_camera_motion((pos_x, pos_y), global_motions)
        adjusted_positions = (adjusted_x, adjusted_y, pos_s)
        # Вставка объекта
        inserter = ObjectInserter(insert_method=insert_method, mask_method=mask_method)
        if insert_method == "poisson":
            inserted_frames_dir = inserter.insert_object_poisson(num_frames, self.raw_frames_path,
                                                                  path_to_object, path_to_mask,
                                                                  adjusted_positions, frame_size, self.save_dir)
        else:
            inserted_frames_dir = inserter.insert_object_simple(num_frames, self.raw_frames_path,
                                                                path_to_object, path_to_mask,
                                                                adjusted_positions, frame_size, self.save_dir)
        # Рендеринг итогового видео
        frames_it = np.arange(num_frames)
        output_video_path = osp.join(self.res_dir, self._get_output_video_name())
        self.render_video(output_video_path, frames_it, inserted_frames_dir, fps)

class ObjectInserter:
    """
    Класс для вставки объекта в кадры видео с использованием двух подходов:
    poisson blending или простая вставка.
    """
    def __init__(self, insert_method: str = "simple", mask_method: str = "auto", poisson_blend_cmd: str = ""):
        self.insert_method = insert_method
        self.mask_method = mask_method
        if poisson_blend_cmd:
            self.poisson_blend_cmd = poisson_blend_cmd
        else:
            self.poisson_blend_cmd = osp.join(os.path.abspath("./poisson_blend/build"), "poisson_blend")

    @log_execution
    def auto_generate_mask(self, path_to_object: str, object_img: np.ndarray) -> np.ndarray:
        """
        Создает маску объекта с помощью GrabCut.
        Объект выделяется красным цветом.
        """
        mask = np.zeros(object_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        h, w = object_img.shape[:2]
        rect = (max(1, int(w * 0.05)), max(1, int(h * 0.05)), w - int(w * 0.1), h - int(h * 0.1))
        cv2.grabCut(object_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype('uint8') * 255
        # цветная маска (выделяем красным)
        mask_colored = cv2.merge([np.zeros_like(mask2), np.zeros_like(mask2), mask2])
        #cv2.imsave("./poisson_blend/img/")
        return mask_colored

    @log_execution
    def insert_object_poisson(self, num_frames: int, raw_path: str, path_to_object: str, path_to_mask: str,
                                positions: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                frame_size: Tuple[int, int], save_dir: str) -> str:
        """
        Вставка объекта с использованием poisson blending.
        """
        insert_path = osp.join(save_dir, "insert")
        os.makedirs(insert_path, exist_ok=True)
        
        object_img = cv2.imread(path_to_object, cv2.IMREAD_UNCHANGED)
        if object_img is None:
            raise ValueError("Не удалось загрузить объект (path_to_object).")
        
        if self.mask_method == "grab":
            mask_img = self.auto_generate_mask(path_to_object, object_img)
        else:
            mask_img = cv2.imread(path_to_mask, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            raise ValueError("Не удалось загрузить или сгенерировать маску.")
        
        x_positions, y_positions, s_positions = positions
        for step in range(num_frames):
            frame_path = osp.join(raw_path, f"frame{step}.png")
            frame = cv2.imread(frame_path)
            if frame is None:
                if osp.exists(frame_path):
                    shutil.copy(frame_path, osp.join(insert_path, f"frame{step}.png"))
                continue
            x = int(x_positions[step])
            y = int(y_positions[step])
            scale_factor = float(s_positions[step])
            obj_h, obj_w = object_img.shape[:2]
            new_w = max(1, int(obj_w * scale_factor))
            new_h = max(1, int(obj_h * scale_factor))
            resized_object = cv2.resize(object_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized_mask = cv2.resize(mask_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_h, frame_w = frame_size
            if x < 0 or y < 0 or (x + new_w) > frame_w or (y + new_h) > frame_h:
                shutil.copy(frame_path, osp.join(insert_path, f"frame{step}.png"))
                continue
            temp_dir = tempfile.mkdtemp()
            temp_object_path = osp.join(temp_dir, 'object.png')
            temp_mask_path = osp.join(temp_dir, 'mask.png')
            temp_frame_path = osp.join(temp_dir, 'frame.png')
            temp_output_path = osp.join(temp_dir, 'output.png')
            cv2.imwrite(temp_object_path, resized_object)
            cv2.imwrite(temp_mask_path, resized_mask)
            cv2.imwrite(temp_frame_path, frame)
            cmd = [
                self.poisson_blend_cmd,
                "-source", temp_object_path,
                "-target", temp_frame_path,
                "-mask", temp_mask_path,
                "-output", temp_output_path,
                "-mx", str(x),
                "-my", str(y)
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Poisson blend не выполнен для кадра {step}: {e}")
            if osp.exists(temp_output_path):
                shutil.copy(temp_output_path, osp.join(insert_path, f"frame{step}.png"))
            else:
                shutil.copy(frame_path, osp.join(insert_path, f"frame{step}.png"))
            shutil.rmtree(temp_dir)
        return insert_path

    @log_execution
    def insert_object_simple(self, num_frames: int, raw_path: str, path_to_object: str, path_to_mask: str,
                               positions: Tuple[np.ndarray, np.ndarray, np.ndarray],
                               frame_size: Tuple[int, int], save_dir: str) -> str:
        """
        Простая вставка объекта без использования poisson blending.
        """
        insert_path = osp.join(save_dir, "insert")
        os.makedirs(insert_path, exist_ok=True)
        
        object_img = cv2.imread(path_to_object, cv2.IMREAD_UNCHANGED)
        if object_img is None:
            raise ValueError("Не удалось загрузить объект (path_to_object).")
        
        if self.mask_method == "grab":
            mask_img = self.auto_generate_mask(path_to_object, object_img)
        else:
            mask_img = cv2.imread(path_to_mask, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            raise ValueError("Не удалось загрузить или сгенерировать маску.")
        
        if len(mask_img.shape) == 3 and mask_img.shape[2] > 1:
            if mask_img.shape[2] == 4:
                mask_gray = mask_img[:, :, 3]
            else:
                mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_img
        
        x_positions, y_positions, s_positions = positions
        for step in range(num_frames):
            frame_path = osp.join(raw_path, f"frame{step}.png")
            frame = cv2.imread(frame_path)
            if frame is None:
                if osp.exists(frame_path):
                    shutil.copy(frame_path, osp.join(insert_path, f"frame{step}.png"))
                continue
            x = int(x_positions[step])
            y = int(y_positions[step])
            scale_factor = float(s_positions[step])
            obj_h, obj_w = object_img.shape[:2]
            new_w = max(1, int(obj_w * scale_factor))
            new_h = max(1, int(obj_h * scale_factor))
            resized_object = cv2.resize(object_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            resized_mask = cv2.resize(mask_gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            frame_h, frame_w = frame_size
            if x < 0 or y < 0 or (x + new_w) > frame_w or (y + new_h) > frame_h:
                shutil.copy(frame_path, osp.join(insert_path, f"frame{step}.png"))
                continue
            roi = frame[y:y+new_h, x:x+new_w]
            if len(resized_object.shape) == 3 and resized_object.shape[2] == 4:
                resized_object = resized_object[:, :, :3]
            mask_bool = resized_mask > 0
            roi[mask_bool] = resized_object[mask_bool]
            frame[y:y+new_h, x:x+new_w] = roi
            out_frame_path = osp.join(insert_path, f"frame{step}.png")
            cv2.imwrite(out_frame_path, frame)
        return insert_path

def main():
    parser = ArgumentParser()
    parser.add_argument("path_to_object", help="Путь к изображению объекта")
    parser.add_argument("path_to_mask", nargs="?", default="",
                        help="Путь к изображению маски (требуется, если выбран режим 'auto')")
    parser.add_argument("--scale", type=float, default=1, help="Масштаб объекта")
    parser.add_argument("--fps", type=int, default=15, help="FPS выходного видео")
    parser.add_argument("--mask_method", choices=["auto", "grab"], default="auto",
                        help="Метод получения маски: 'auto' — загрузить маску из файла, 'grab' — сгенерировать с помощью GrabCut")
    parser.add_argument("--insert_method", choices=["poisson", "simple"], default="simple",
                        help="Метод вставки объекта: 'poisson' — с использованием poisson_blend, 'simple' — простая вставка")
    args = parser.parse_args()

    if args.mask_method == "auto" and not args.path_to_mask:
        parser.error("При выборе режима 'auto' для маски необходимо указать путь к маске.")

    VIDS_DIR = "vids"
    video_files = [f for f in os.listdir(VIDS_DIR)
                   if osp.isfile(osp.join(VIDS_DIR, f))
                   and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not video_files:
        print("Нет видеофайлов для обработки.")
        return
    for video_name in video_files:
        video_path = osp.join(VIDS_DIR, video_name)
        processor = VideoProcessor(video_path)
        processor.process_video(args.path_to_object, args.path_to_mask, args.fps,
                                args.mask_method, args.insert_method)

if __name__ == "__main__":
    main()
