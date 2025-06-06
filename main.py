import cv2 as cv
import numpy as np
import imageio
import os
import shutil

# GIF → PNG 시퀀스로 변환
def gif_to_png_sequence(gif_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gif = imageio.mimread(gif_path)
    
    for i, frame in enumerate(gif):
        frame = np.array(frame)

        if frame.shape[2] == 3:
            # 알파 채널 추가
            alpha = np.ones((frame.shape[0], frame.shape[1], 1), dtype=np.uint8) * 255
            frame = np.concatenate((frame, alpha), axis=2)

        # 첫 프레임이라면, 검정 배경 픽셀을 투명하게 만듬
        if i == 0:
            mask = np.all(frame[:, :, :3] == [0, 0, 0], axis=2)
            frame[mask, 3] = 0  # 알파 채널을 0으로 설정 (투명)

        imageio.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), frame)

# PNG 프레임 불러오기
def load_png_sequence(folder):
    filenames = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    frames = []
    for name in filenames:
        img = cv.imread(os.path.join(folder, name), cv.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.shape[2] == 3:
            alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
            img = np.concatenate((img, alpha), axis=2)
        frames.append(img)
    return frames

# pngs 디렉토리에 파일 초기화화
def remove_png_files(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    else:
        pass

# 카메라 보정
def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    img_points = []
    gray = None
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, pts = cv.findChessboardCorners(gray, board_pattern)
        if ret:
            img_points.append(pts)
    assert len(img_points) > 0, "체커보드가 감지된 프레임이 없습니다"

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)

    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

# 설정
chessboard_video_name = "chessboard.avi"
gif_path = "example.gif"
png_dir = "./pngs"
board_pattern = (10, 7)
cellsize = 25.0
start_x, start_y = 2, 1
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
gif_frame_delay = 3

# GIF → PNG 변환
remove_png_files(png_dir)
gif_to_png_sequence(gif_path, png_dir)
png_frames = load_png_sequence(png_dir)

# 비디오 설정
cap = cv.VideoCapture(chessboard_video_name)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# 카메라 보정용 프레임 수집
chessboard_frames = []
for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, _ = cv.findChessboardCorners(gray, board_pattern)
    if found:
        chessboard_frames.append(frame)
    if len(chessboard_frames) >= 10:
        break

# 보정 카메라 파라메터 수집
if len(chessboard_frames) > 0:
    ret, K, dist_coeffs, _, _ = calib_camera_from_chessboard(
        chessboard_frames, board_pattern, cellsize)

# 체스보드 3D 모델 좌표
objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
objp[:, :2] = np.indices(board_pattern).T.reshape(-1, 2)
objp *= cellsize

# 영상 다시 처음부터
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, board_pattern) # 코너 찾기

    if found:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),criteria) # 코너 검출 정확도 높이기
        success, rvec, tvec = cv.solvePnP(objp, corners, K, dist_coeffs)


        # 변환한 PNG를 순서대로 나타나게 하여 GIF 처럼 보이게 하기
    if success:
        if frame_idx % gif_frame_delay == 0:
            gif_index = (frame_idx//gif_frame_delay) % len(png_frames)

        overlay_frame = png_frames[gif_index]
        frame_idx += 1

            # GIF 크기 조절
        gif_h, gif_w = overlay_frame.shape[:2]
        aspect_ratio = gif_w / gif_h
        cell_width = 3
        cell_height = cell_width / aspect_ratio
        overlay_resized = cv.resize(overlay_frame, (gif_w, gif_h))
        # overlay_resized = cv.flip(overlay_resized, 0)
        gif_w_mm = cell_width * cellsize
        gif_h_mm = cell_height * cellsize

        # GIF 나타나게 할 체스보드 좌표
        model_pts = np.float32([
        [0, 0, -gif_h_mm],          # top-left
        [0, 0, 0],                  # bottom_left
        [gif_w_mm, 0, -gif_h_mm],   # top-right
        [gif_w_mm, 0, 0]            # bottom-right
        ]) + np.float32([start_x * cellsize, start_y * cellsize, 0])


        img_pts, _ = cv.projectPoints(model_pts, rvec, tvec, K, dist_coeffs)
        dst_pts = img_pts.reshape(-1, 2).astype(np.float32)

            # 순서 맞춰서 src_pts도 일치시킴
        src_pts = np.float32([
            [0, 0],        # top-left
            [0, gif_h],    # bottom-left
            [gif_w, 0],    # top-right
            [gif_w, gif_h] # bottom-right
        ])
        # 투시 변환 행렬 구하기
        M = cv.getPerspectiveTransform(src_pts, dst_pts)

            # src 이미지에서 dst 이미지로 픽셀을 재배치
        warped = cv.warpPerspective(overlay_resized, M, (frame.shape[1], frame.shape[0]),
                                        borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        alpha_channel = warped[:, :, 3]
        _, mask_binary = cv.threshold(alpha_channel, 1, 255, cv.THRESH_BINARY)

        overlay_bgr = warped[:, :, :3].astype(np.uint8)
        mask_binary = mask_binary.astype(np.uint8)

        cv.copyTo(src=overlay_bgr, dst=frame, mask=mask_binary)

    out.write(frame)

cap.release()
out.release()
cv.destroyAllWindows()
