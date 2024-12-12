import numpy as np
import os
import os.path as osp
import cv2
import shutil
from simulate_trajectory import simulate_traj as simtr


def parse_video(path, out):
    vidcap = cv2.VideoCapture(path)
    success, frame = vidcap.read()
    frame_size = [0, 0]
    count = 0

    while success:
        name = osp.join(out, f"frame{count}.png")
        frame_size = frame.shape
        if not osp.exists(name):
            cv2.imwrite(name, frame)
        success, frame = vidcap.read()
        count += 1
    return count, frame_size


def render_video(output_video_path, frames, frames_dir, fps):
    filename = osp.join(frames_dir, f"frame{frames[0]}.png")
    img = cv2.imread(filename)
    frameSize = (img.shape[1], img.shape[0])
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frameSize
    )
    print(f"Сохранение видео в {output_video_path}")
    for i in frames:
        filename = osp.join(frames_dir, f"frame{i}.png")
        if osp.exists(filename):
            img = cv2.imread(filename)
            frame = cv2.resize(img, frameSize)
            out.write(frame)
    out.release()


def get_frames(path_to_video, path_to_save, scale=1):
    raw_path = osp.join(path_to_save, "raw_frames")
    if not osp.exists(raw_path):
        os.mkdir(raw_path)
    num_frames, frame_size = parse_video(path_to_video, raw_path)
    return num_frames, raw_path, frame_size


def cut_video_frames(path_to_video, path_to_save, scale=1):
    if not osp.exists(path_to_save):
        os.mkdir(path_to_save)
    num_frames, raw_path, frame_size = get_frames(
        path_to_video, path_to_save, scale
    )
    print(f"Количество кадров: {num_frames}")
    return num_frames, raw_path, frame_size


def crop_to_content(img, mask):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_img = img[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        print(x, y, w, h)
        return cropped_img, cropped_mask
    else:
        return img, mask

def insert_object(num_frames, raw_path, path_to_object, path_to_mask, path_to_save, positions, object_size, frame_size):
    insert_path = osp.join(path_to_save, "insert")

    if not osp.exists(insert_path):
        os.mkdir(insert_path)

    cmd_base = os.path.abspath("./poisson_blend/build/poisson_blend")
    print(f"Позиции вставки объекта: {positions}")
    object_width, object_height = object_size[1], object_size[0]
    frame_width, frame_height = frame_size[1], frame_size[0]

    for step in range(0, num_frames):
        name = osp.join(insert_path, f"frame{step}.png")
        if osp.exists(name):
            continue

        mx = int(positions[0][step])
        my = int(positions[1][step])

        frame_path = osp.join(raw_path, f"frame{step}.png")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Не удалось загрузить кадр {step} по пути: {frame_path}")
            continue

        obj_img = cv2.imread(path_to_object, cv2.IMREAD_UNCHANGED)
        if obj_img is None:
            print(f"Не удалось загрузить изображение объекта по пути: {path_to_object}")
            continue

        mask_img = cv2.imread(path_to_mask, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            print(f"Не удалось загрузить маску по пути: {path_to_mask}")
            continue

        x_start = max(mx, 0)
        y_start = max(my, 0)
        x_end = min(mx + object_width, frame_width)
        y_end = min(my + object_height, frame_height)
        obj_x_start = max(-mx, 0)
        obj_y_start = max(-my, 0)
        obj_x_end = obj_x_start + (x_end - x_start)
        obj_y_end = obj_y_start + (y_end - y_start)

        if x_start >= x_end or y_start >= y_end:
            print(f"Кадр {step}: объект полностью за пределами кадра. Сохранение исходного кадра без изменений.")
            shutil.copy(frame_path, name)
            continue

        obj_crop = obj_img[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
        mask_crop = mask_img[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
        if obj_crop.shape[0] < 1 or obj_crop.shape[1] < 1:
            print(f"Кадр {step}: обрезанное изображение объекта слишком мало. Сохранение исходного кадра без изменений.")
            shutil.copy(frame_path, name)
            continue

        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_object_path = osp.join(temp_dir, 'cropped_object.png')
        temp_mask_path = osp.join(temp_dir, 'cropped_mask.png')
        temp_frame_path = osp.join(temp_dir, 'cropped_frame.png')
        temp_output_path = osp.join(temp_dir, 'output.png')

        cv2.imwrite(temp_object_path, obj_crop)
        cv2.imwrite(temp_mask_path, mask_crop)
        cv2.imwrite(temp_frame_path, frame)
        new_mx = x_start
        new_my = y_start

        if new_mx <= 0:
            new_mx = 1
        if new_my <= 0:
            new_my = 1

        source = ' -source "' + temp_object_path + '"'
        target = ' -target "' + temp_frame_path + '"'
        mask = ' -mask "' + temp_mask_path + '"'
        mx_str = " -mx " + str(new_mx)
        my_str = " -my " + str(new_my)
        output = ' -output "' + temp_output_path + '"'

        cmd = cmd_base + source + target + mask + output + mx_str + my_str
        print(f"Выполнение команды: {cmd}")
        os.system(cmd)

        if not osp.exists(temp_output_path):
            print(f"Кадр {step}: не удалось создать выходной файл. Сохранение исходного кадра без изменений.")
            shutil.copy(frame_path, name)
            shutil.rmtree(temp_dir)
            continue

        shutil.copy(temp_output_path, name)
        shutil.rmtree(temp_dir)
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

    num_frames, raw_path, frame_size = cut_video_frames(
        path_to_video, path_to_save, scale
    )

    object_img = cv2.imread(path_to_object, cv2.IMREAD_UNCHANGED)
    mask_img = cv2.imread(path_to_mask, cv2.IMREAD_UNCHANGED)

    if object_img is None:
        print(f"Не удалось загрузить изображение объекта по пути: {path_to_object}")
        return
    if mask_img is None:
        print(f"Не удалось загрузить маску по пути: {path_to_mask}")
        return

    object_img_cropped, mask_img_cropped = crop_to_content(object_img, mask_img)
    object_size = object_img_cropped.shape  # (height, width, channels)
    object_height, object_width = object_size[:2]
    frame_height, frame_width = frame_size[0], frame_size[1]
    print(f"Размер кадра: {frame_width}x{frame_height}")
    print(f"Размер объекта после обрезки: {object_width}x{object_height}")

    if object_width > frame_width - 2 or object_height > frame_height - 2:
        print("Размер объекта слишком велик. Пожалуйста, используйте объект меньшего размера или уменьшите объект.")
        return

    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_object_path = osp.join(temp_dir, 'cropped_object.png')
    temp_mask_path = osp.join(temp_dir, 'cropped_mask.png')
    cv2.imwrite(temp_object_path, object_img_cropped)
    cv2.imwrite(temp_mask_path, mask_img_cropped)

    positions = np.array(
        simtr(num_frames, (frame_height, frame_width), object_size)
    )
    insert_path = insert_object(
        num_frames, raw_path, temp_object_path, temp_mask_path, path_to_save, positions, object_size, frame_size
    )

    frames_it = np.arange(0, num_frames, 1)

    output_video_name = osp.splitext(video_name)[0] + "+obj.mp4"
    output_video_path = osp.join(RES_DIR, output_video_name)
    render_video(output_video_path, frames_it, insert_path, fps)
    shutil.rmtree(temp_dir)


def main(path_to_object, path_to_mask, scale=1, fps=15):
    VIDS_DIR = "vids"
    video_files = [
        f
        for f in os.listdir(VIDS_DIR)
        if osp.isfile(osp.join(VIDS_DIR, f))
        and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print(f"Нет видеофайлов в папке {VIDS_DIR}")
        return 0

    for video_name in video_files:
        print(f"Обработка видео: {video_name}")
        process_video(video_name, path_to_object, path_to_mask, scale, fps)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("path_to_object")
    parser.add_argument("path_to_mask")
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--fps", type=int, default=15, help="Частота кадров итогового видео")

    args = parser.parse_args()

    main(args.path_to_object, args.path_to_mask, args.scale, args.fps)




