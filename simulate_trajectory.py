import numpy as np

def simulate_traj(n_frames, frame_size, object_size):
    input_file = 'input.txt'
    key_frames = []
    with open(input_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 3:
                frame_number = int(tokens[0])
                x_coordinate = float(tokens[1])
                y_coordinate = float(tokens[2])
                key_frames.append((frame_number, x_coordinate, y_coordinate))
            else:
                print(f"Неверный формат строки в input.txt: {line}")
    
    if not key_frames:
        print("Траектория не задана в input.txt")
        return None

    key_frames.sort(key=lambda x: x[0])
    positions_x = np.zeros(n_frames)
    positions_y = np.zeros(n_frames)
    first_frame, first_x, first_y = key_frames[0]
    if first_frame > 0:
        positions_x[:first_frame] = first_x
        positions_y[:first_frame] = first_y

    for i in range(len(key_frames) - 1):
        frame_start, x_start, y_start = key_frames[i]
        frame_end, x_end, y_end = key_frames[i+1]
        frames_range = frame_end - frame_start
        if frames_range <= 0:
            print(f"Неверный диапазон между кадрами {frame_start} и {frame_end}")
            continue
        t = np.linspace(0, 1, frames_range + 1)
        positions_x[frame_start:frame_end+1] = x_start + (x_end - x_start) * t
        positions_y[frame_start:frame_end+1] = y_start + (y_end - y_start) * t

    last_frame, last_x, last_y = key_frames[-1]
    if last_frame < n_frames - 1:
        positions_x[last_frame:] = last_x
        positions_y[last_frame:] = last_y

    if len(key_frames) == 1:
        positions_x[:] = first_x
        positions_y[:] = first_y

    return positions_x, positions_y