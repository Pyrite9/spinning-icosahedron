import sys
import os
import time
import numpy as np

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
phi         = (1 + np.sqrt(5)) / 2         # 황금비 ≈ 1.618
CAMERA_DIST = 5                             # 카메라 Z축 오프셋
CIRCUM_R    = float(np.sqrt(1 + phi ** 2)) # 정이십면체 외접구 반지름 ≈ 1.902

# ──────────────────────────────────────────────
# 화면 설정
# ──────────────────────────────────────────────

def get_terminal_size():
    """터미널 크기와 투영 배율을 반환. 감지 실패 시 기본값 사용."""
    try:
        term_size = os.get_terminal_size()
        w = term_size.columns
        h = term_size.lines - 1  # 프롬프트 줄 하나 빼기
    except OSError:
        w, h = 80, 40
    scale = h * 0.375 * CAMERA_DIST / (phi * 0.5)
    return w, h, scale

WIDTH, HEIGHT, PROJ_SCALE = get_terminal_size()

# ──────────────────────────────────────────────
# 정이십면체 꼭짓점 정의
# 황금비(φ)를 이용한 표준 좌표계
# 12개 꼭짓점: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)의 짝수 치환
# ──────────────────────────────────────────────
vertices = np.array([
    [ 0,  1,  phi],  # 0
    [ 0,  1, -phi],  # 1
    [ 0, -1,  phi],  # 2
    [ 0, -1, -phi],  # 3
    [ 1,  phi,  0],  # 4
    [ 1, -phi,  0],  # 5
    [-1,  phi,  0],  # 6
    [-1, -phi,  0],  # 7
    [ phi,  0,  1],  # 8
    [ phi,  0, -1],  # 9
    [-phi,  0,  1],  # 10
    [-phi,  0, -1],  # 11
], dtype=float)

# ──────────────────────────────────────────────
# 정이십면체 면 정의 (20개 삼각형)
# 꼭짓점 순서는 바깥쪽에서 봤을 때 반시계(CCW) 기준
# ──────────────────────────────────────────────
faces = [
    # 꼭짓점 0 주변 상단 캡 (5개)
    (0, 2, 8),
    (0, 8, 4),
    (0, 4, 6),
    (0, 6, 10),
    (0, 10, 2),

    # 꼭짓점 3 주변 하단 캡 (5개)
    (3, 1, 9),
    (3, 9, 5),
    (3, 5, 7),
    (3, 7, 11),
    (3, 11, 1),

    # 중간 띠 (10개)
    (8, 2, 5),
    (8, 5, 9),
    (8, 9, 4),
    (4, 9, 1),
    (4, 1, 6),
    (6, 1, 11),
    (6, 11, 10),
    (10, 11, 7),
    (10, 7, 2),
    (2, 7, 5),
]

# ──────────────────────────────────────────────
# 셰이딩용 ASCII 문자 (어두움 → 밝음)
# ──────────────────────────────────────────────
shade_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# 광원 방향 (정규화된 단위 벡터, 우상단 대각선)
light_dir = np.array([1, 1, 1]) / np.sqrt(3)


# ──────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────

def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotation_matrix_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def project(point):
    """
    3D 좌표 → 2D 화면 좌표로 원근 투영.
    반환: (screen_x, screen_y) 또는 카메라 뒤에 있으면 (None, None)
    """
    z = point[2] + CAMERA_DIST
    if z <= 0:
        return None, None
    # Y축에 0.5를 곱하는 이유: 콘솔 문자가 세로로 길어서 종횡비 보정
    x = int(point[0] * PROJ_SCALE / z + WIDTH / 2)
    y = int(-point[1] * PROJ_SCALE / z * 0.5 + HEIGHT / 2)
    return x, y


def draw_line(frame, x0, y0, x1, y1, char='*'):
    """Bresenham 직선 알고리즘으로 두 점 사이에 문자를 찍음."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < WIDTH and 0 <= y0 < HEIGHT:
            frame[y0][x0] = char
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def is_face_visible(v0, v1, v2):
    """
    실루엣 기반 후면 컬링 (2D 스크린 공간 와인딩 판별).

    투영된 삼각형의 부호 면적으로 전/후면을 결정:
    - 부호 면적 > 0 : 반시계(CCW) → 전면 (visible)
    - 부호 면적 ≤ 0 : 시계(CW) 또는 퇴화 → 후면 (cull)

    실루엣 경계가 CCW/CW 전환점이 되며, 투시 투영 후 2D에서 직접 판별하므로
    법선 근사 오류 없이 정확하게 동작한다.
    """
    if v0[2] + CAMERA_DIST <= 0 and v1[2] + CAMERA_DIST <= 0 and v2[2] + CAMERA_DIST <= 0:
        return False

    x0, y0 = project(v0)
    x1, y1 = project(v1)
    x2, y2 = project(v2)

    if x0 is None or x1 is None or x2 is None:
        return False

    # 2D 부호 면적: (v1-v0) × (v2-v0)
    signed_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    return signed_area > 0


def fill_triangle(frame, z_buffer, v0, v1, v2):
    """
    삼각형 래스터라이저 (flat shading + Z-buffer, NumPy 벡터화).

    1. 세 꼭짓점을 2D로 투영
    2. 면의 법선으로 광원과의 각도 계산 → 밝기 결정
    3. Bounding box 내 픽셀을 NumPy 배열로 한 번에 처리:
       - 무게중심 좌표 벡터 연산으로 삼각형 내부 마스킹
       - Z-buffer 비교 후 더 가까운 픽셀만 문자 갱신
    """
    x0, y0 = project(v0)
    x1, y1 = project(v1)
    x2, y2 = project(v2)

    if None in (x0, y0, x1, y1, x2, y2):
        return

    # Bounding box (화면 범위로 클리핑)
    min_x = max(0, min(x0, x1, x2))
    max_x = min(WIDTH - 1, max(x0, x1, x2))
    min_y = max(0, min(y0, y1, y2))
    max_y = min(HEIGHT - 1, max(y0, y1, y2))

    if min_x > max_x or min_y > max_y:
        return

    # 면 법선 벡터 → 라이팅
    normal = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-10:
        return
    normal /= norm_len

    # 법선이 항상 바깥쪽을 향하도록 보정
    center = (v0 + v1 + v2) / 3
    if np.dot(normal, center) < 0:
        normal = -normal

    # Lambert 70% + 깊이 30% 혼합
    lambert     = max(0.0, np.dot(normal, light_dir))
    depth_score = max(0.0, min(1.0, (CIRCUM_R - center[2]) / (2.0 * CIRCUM_R)))
    combined    = 0.7 * lambert + 0.3 * depth_score
    char        = shade_chars[int(combined * (len(shade_chars) - 1))]

    # ── NumPy 벡터화 래스터라이즈 ──
    ys, xs = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-10:
        return

    w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
    w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
    w2 = 1 - w0 - w1

    inside  = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    z_vals  = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

    # Z-buffer 비교: 내부이고 더 가까운 픽셀만 갱신
    update_mask = inside & (z_vals < z_buffer[min_y:max_y + 1, min_x:max_x + 1])
    z_buffer[min_y:max_y + 1, min_x:max_x + 1][update_mask] = z_vals[update_mask]
    frame[min_y:max_y + 1, min_x:max_x + 1][update_mask] = char


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────

# ANSI 이스케이프 시퀀스
# \033[H    : 커서를 (0,0)으로 이동 (화면을 지우지 않고 덮어쓰기)
# \033[2J   : 화면 전체 지우기 (최초 1회만 사용)
# \033[?25l : 커서 숨기기
# \033[?25h : 커서 보이기

angle_x = 0.0
angle_y = 0.0
angle_z = 0.0

sys.stdout.write('\033[2J\033[?25l')
sys.stdout.flush()

try:
    while True:
        WIDTH, HEIGHT, PROJ_SCALE = get_terminal_size()
        sys.stdout.write('\033[H')

        # ── 1. 회전 행렬 합성 (Z → Y → X 순) ──
        R = rotation_matrix_z(angle_z) @ rotation_matrix_y(angle_y) @ rotation_matrix_x(angle_x)
        rotated = vertices @ R.T

        # ── 2. 프레임 버퍼 & Z-버퍼 초기화 ──
        frame    = np.full((HEIGHT, WIDTH), ' ', dtype='U1')
        z_buffer = np.full((HEIGHT, WIDTH), np.inf)

        # ── 3. 가시성 판별 & 면 채우기 ──
        # 결과를 캐싱하여 엣지 단계에서 재사용
        face_visible = []
        for i, j, k in faces:
            v0, v1, v2 = rotated[i], rotated[j], rotated[k]
            visible = is_face_visible(v0, v1, v2)
            face_visible.append(visible)
            if visible:
                fill_triangle(frame, z_buffer, v0, v1, v2)

        # ── 4. 엣지 그리기 (실루엣: *, 내부선: -) ──
        edge_visibility = {}  # (i,j) → 인접 면 가시성 집합 {True, False}
        for (i, j, k), visible in zip(faces, face_visible):
            for a, b in ((i, j), (j, k), (k, i)):
                edge_visibility.setdefault((min(a, b), max(a, b)), set()).add(visible)

        for (i, j), vis_set in edge_visibility.items():
            x0, y0 = project(rotated[i])
            x1, y1 = project(rotated[j])
            if x0 is None or x1 is None:
                continue
            if True in vis_set and False in vis_set:
                draw_line(frame, x0, y0, x1, y1, '*')  # 실루엣
            elif True in vis_set:
                draw_line(frame, x0, y0, x1, y1, '-')  # 내부선

        # ── 5. 화면 출력 ──
        # 전체 프레임을 하나의 문자열로 만들어 한 번에 write (깜빡임 방지)
        sys.stdout.write('\n'.join(''.join(row) for row in frame.tolist()))
        sys.stdout.flush()

        # ── 6. 각도 업데이트 (각 축 다른 속도로 회전) ──
        angle_x += 0.03
        angle_y += 0.02
        angle_z += 0.015

        time.sleep(0.05)

except KeyboardInterrupt:
    # Ctrl+C로 종료 시 커서 복원 + 화면 정리
    sys.stdout.write('\033[?25h\033[2J\033[H')
    sys.stdout.flush()
