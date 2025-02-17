import numpy as np
import os
import os.path as osp
import cv2
import shutil
from simulate_trajectory import simulate_traj as simtr

def parse_video(path, out):
    vidcap = cv2.VideoCapture(path)
    success, frame = vidcap.read()
    frame_size = None
    count = 0

    if not osp.exists(out):
        os.makedirs(out)

    while success:
        name = osp.join(out, f"frame{count}.png")
        if frame is not None:
            frame_size = frame.shape[:2]  # Only height and width
            cv2.imwrite(name, frame)
        else:
            print(f"Warning: Frame {count} is empty and will be skipped.")
        success, frame = vidcap.read()
        count += 1

    if count == 0:
        raise ValueError("No frames extracted. Please check the video file.")

    print(f"Extracted {count} frames to {out}")
    return count, frame_size

def render_video(output_video_path, frames, frames_dir, fps):
    first_frame_path = osp.join(frames_dir, f"frame{frames[0]}.png")
    img = cv2.imread(first_frame_path)

    if img is None:
        raise ValueError(f"First frame not found or empty: {first_frame_path}")

    frameSize = (img.shape[1], img.shape[0])
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frameSize
    )
    print(f"Saving video to {output_video_path}")

    for i in frames:
        filename = osp.join(frames_dir, f"frame{i}.png")
        if osp.exists(filename):
            img = cv2.imread(filename)
            if img is not None:
                frame = cv2.resize(img, frameSize)
                out.write(frame)
            else:
                print(f"Warning: Frame {i} is empty. Skipping.")
        else:
            print(f"Warning: Frame {i} does not exist. Skipping.")
    out.release()

from pykalman import KalmanFilter

def compute_camera_motion(raw_frames_path, num_frames, threshold=0.01):
    """
    Теперь возвращаем список из num_frames матриц 3×3.
    motions[i] -- переход из (i-1)-го кадра в i-й (affine).
    Для i=0 берём единичную матрицу.
    """
    motions = []
    prev_frame = None

    for i in range(num_frames):
        frame_path = osp.join(raw_frames_path, f"frame{i}.png")
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        if frame is None:
            print(f"Warning: Frame {i} is missing or empty. Using identity motion.")
            motions.append(np.eye(3))
            continue

        if prev_frame is None:
            prev_frame = frame
            motions.append(np.eye(3))  # no motion for frame 0
            continue

        detector = cv2.AKAZE_create()
        kp1, des1 = detector.detectAndCompute(prev_frame, None)
        kp2, des2 = detector.detectAndCompute(frame, None)

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
                        motions.append(np.eye(3))
                else:
                    motions.append(np.eye(3))
            else:
                motions.append(np.eye(3))
        else:
            motions.append(np.eye(3))

        prev_frame = frame

    return motions

def smooth_motion(motions):
    """
    Если захотите сглаживать, можно оставить как было —
    но нужно, чтобы все motions[i] были 3×3.
    Ниже упрощённый пример сглаживания только транслейшна.
    """
    translations = np.array([m[:2, 2] for m in motions])  # (n, 2)
    kf = KalmanFilter(initial_state_mean=np.zeros(2), n_dim_obs=2)
    kf = kf.em(translations, n_iter=25)
    smoothed_states, _ = kf.smooth(translations)

    smoothed_motions = []
    for i, motion in enumerate(motions):
        M = motion.copy()
        M[:2, 2] = smoothed_states[i]
        smoothed_motions.append(M)
    return smoothed_motions

def accumulate_motions(motions):
    """
    Переводим список относительных матриц motions[i] (переход (i-1)->i)
    в "глобальные" матрицы: global_motions[i] (переход 0->i).
    """
    num_frames = len(motions)
    global_motions = [np.eye(3, dtype=np.float32) for _ in range(num_frames)]

    for i in range(1, num_frames):
        global_motions[i] = motions[i] @ global_motions[i - 1]

    return global_motions

def project_positions_with_camera_motion(positions, global_motions):
    """
    positions: (x_array, y_array) длины n_frames.
    global_motions[i]: матрица 3×3, переход (0)->i.

    Возвращаем "камерные" координаты (adjusted_x, adjusted_y) для кадра i.
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

def insert_object(num_frames, raw_path, path_to_object, path_to_mask, positions, frame_size, path_to_save):
    insert_path = osp.join(path_to_save, "insert")
    if not osp.exists(insert_path):
        os.makedirs(insert_path)

    object_img = cv2.imread(path_to_object, cv2.IMREAD_UNCHANGED)
    mask_img = cv2.imread(path_to_mask, cv2.IMREAD_UNCHANGED)

    if object_img is None or mask_img is None:
        raise ValueError("Object or mask image could not be loaded.")

    cmd_base = os.path.abspath("./poisson_blend/build/poisson_blend")

    for step in range(num_frames):
        frame_path = osp.join(raw_path, f"frame{step}.png")
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: Frame {step} is missing or empty. Skipping.")
            continue

        x, y = int(positions[0][step]), int(positions[1][step])
        obj_h, obj_w = object_img.shape[:2]
        frame_h, frame_w = frame_size

        # Проверяем, что объект не вылезает за границы
        if x < 0 or y < 0 or x + obj_w > frame_w or y + obj_h > frame_h:
            print(f"Object at frame {step} does not fit in frame. Skipping.")
            shutil.copy(frame_path, osp.join(insert_path, f"frame{step}.png"))
            continue

        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_object_path = osp.join(temp_dir, 'object.png')
        temp_mask_path = osp.join(temp_dir, 'mask.png')
        temp_frame_path = osp.join(temp_dir, 'frame.png')
        temp_output_path = osp.join(temp_dir, 'output.png')

        cv2.imwrite(temp_object_path, object_img)
        cv2.imwrite(temp_mask_path, mask_img)
        cv2.imwrite(temp_frame_path, frame)

        cmd = f"{cmd_base} -source {temp_object_path} -target {temp_frame_path} -mask {temp_mask_path} -output {temp_output_path} -mx {x} -my {y}"
        os.system(cmd)

        if osp.exists(temp_output_path):
            shutil.copy(temp_output_path, osp.join(insert_path, f"frame{step}.png"))
        else:
            print(f"Failed to blend object at frame {step}. Saved original frame.")
            shutil.copy(frame_path, osp.join(insert_path, f"frame{step}.png"))

        shutil.rmtree(temp_dir)

    print(f"Inserted object into frames using Poisson blending and saved to {insert_path}")
    return insert_path

def process_video(video_name, path_to_object, path_to_mask, scale=1, fps=15):
    VIDS_DIR = "vids"
    RES_DIR = "res"
    SAVE_DIR = "save"
    path_to_video = osp.join(VIDS_DIR, video_name)
    path_to_save = SAVE_DIR

    if osp.exists(path_to_save):
        shutil.rmtree(path_to_save)
    os.makedirs(path_to_save)

    if not osp.exists(RES_DIR):
        os.makedirs(RES_DIR)

    # 1. Извлекаем кадры
    num_frames, frame_size = parse_video(path_to_video, osp.join(path_to_save, "raw_frames"))

    # 2. Считаем относительные движения
    motions = compute_camera_motion(osp.join(path_to_save, "raw_frames"), num_frames)

    # (Можно при желании сгладить:
    #motions = smooth_motion(motions)
    # )

    # 3. Аккумулируем их в глобальные
    global_motions = accumulate_motions(motions)

    # 4. Получаем траекторию "в мире"
    original_positions = simtr(num_frames)
      # original_positions[0] — x-коорд. для каждого кадра
      # original_positions[1] — y-коорд. для каждого кадра

    # 5. Проецируем в координаты кадра
    adjusted_x, adjusted_y = project_positions_with_camera_motion(original_positions, global_motions)
    adjusted_positions = (adjusted_x, adjusted_y)

    # 6. Вставляем объект
    insert_path = insert_object(
        num_frames,
        osp.join(path_to_save, "raw_frames"),
        path_to_object,
        path_to_mask,
        adjusted_positions,
        frame_size,
        path_to_save
    )

    # 7. Рендерим финальное видео
    frames_it = np.arange(0, num_frames, 1)
    output_video_name = osp.splitext(video_name)[0] + "_synced.mp4"
    output_video_path = osp.join(RES_DIR, output_video_name)
    render_video(output_video_path, frames_it, insert_path, fps)

def main(path_to_object, path_to_mask, scale=1, fps=15):
    VIDS_DIR = "vids"
    video_files = [
        f
        for f in os.listdir(VIDS_DIR)
        if osp.isfile(osp.join(VIDS_DIR, f))
        and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print(f"No video files in folder {VIDS_DIR}")
        return 0

    for video_name in video_files:
        print(f"Processing video: {video_name}")
        process_video(video_name, path_to_object, path_to_mask, scale, fps)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("path_to_object")
    parser.add_argument("path_to_mask")
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--fps", type=int, default=15, help="Frame rate of the output video")
    args = parser.parse_args()
    main(args.path_to_object, args.path_to_mask, args.scale, args.fps)
