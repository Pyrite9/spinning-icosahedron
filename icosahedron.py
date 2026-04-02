import numpy as np
import time
import os

# ──────────────────────────────────────────────
# 화면 설정
# 터미널 크기를 자동 감지. 감지 실패 시 기본값 사용.
# ──────────────────────────────────────────────
try:
    term_size = os.get_terminal_size()
    WIDTH = term_size.columns
    HEIGHT = term_size.lines - 1  # 프롬프트 줄 하나 빼기
except OSError:
    WIDTH = 80
    HEIGHT = 40

# ──────────────────────────────────────────────
# 정이십면체 꼭짓점 정의
# 황금비(φ)를 이용한 표준 좌표계
# 12개 꼭짓점: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)의 짝수 치환
# ──────────────────────────────────────────────
phi = (1 + np.sqrt(5)) / 2  # 황금비 ≈ 1.618

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
], dtype=float)  # 스케일은 투영 단계에서 처리

# ──────────────────────────────────────────────
# 정이십면체 면 정의 (20개 삼각형)
#
# ──────────────────────────────────────────────
faces = [
    # 꼭짓점 0 주변 상단 캡 (5개)
    (0, 2, 8),
    (0, 8, 4),
    (0, 4, 6),
    (0, 6, 10),
    (0, 10, 2),

    # 꼭짓점 3 주변 하단 캡 (5개)
    (3, 9, 1),
    (3, 5, 9),
    (3, 7, 5),
    (3, 11, 7),
    (3, 1, 11),

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
# 인덱스가 클수록 밝은 문자
# ──────────────────────────────────────────────
shade_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# 광원 방향 (정규화된 단위 벡터, 우상단 대각선)
light_dir = np.array([1, 1, 1]) / np.sqrt(3)

# 카메라까지의 거리 (Z축 오프셋)
CAMERA_DIST = 5

# 투영 배율 — 화면 높이 기준으로 이십면체가 약 75%를 차지하도록 계산.
# 정이십면체 꼭짓점의 최대 반경 = phi ≈ 1.618
# 화면에 투영될 크기 = PROJ_SCALE * phi / CAMERA_DIST * 0.5 (종횡비 보정)
# 이것이 HEIGHT * 0.375 (= 75% / 2, 상하 대칭)가 되도록 역산.
PROJ_SCALE = HEIGHT * 0.375 * CAMERA_DIST / ((1 + np.sqrt(5)) / 2 * 0.5)


# ──────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────

def rotation_matrix_x(angle):
    """X축 회전 행렬"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

def rotation_matrix_y(angle):
    """Y축 회전 행렬"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(angle):
    """Z축 회전 행렬"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


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
    """
    Bresenham 직선 알고리즘으로 두 점 사이에 문자를 찍음.
    와이어프레임 엣지 렌더링에 사용.
    """
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


def barycentric(px, py, x0, y0, x1, y1, x2, y2):
    """
    무게중심 좌표(Barycentric Coordinates) 계산.
    
    점 (px, py)가 삼각형 (x0,y0)-(x1,y1)-(x2,y2) 내부에 있는지 판별하고,
    내부라면 각 꼭짓점에 대한 가중치를 반환.
    
    반환: (w0, w1, w2)
      - 모두 >= 0이면 삼각형 내부
      - 하나라도 < 0이면 외부
    """
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

    if abs(denom) < 1e-10:  # 퇴화 삼각형 (세 점이 일직선)
        return -1, -1, -1

    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1 - w0 - w1

    return w0, w1, w2


def is_face_visible(v0, v1, v2):
    """
    면 가시성 판별.
    Z-buffer가 앞/뒤 면 겹침을 처리하므로,
    카메라 뒤로 완전히 넘어간 경우만 걸러냄.
    """
    # 세 꼭짓점 모두 카메라 뒤에 있으면 스킵
    if v0[2] + CAMERA_DIST <= 0 and v1[2] + CAMERA_DIST <= 0 and v2[2] + CAMERA_DIST <= 0:
        return False
    return True


def fill_triangle(frame, z_buffer, v0, v1, v2):
    """
    삼각형 래스터라이저 (flat shading + Z-buffer).
    
    1. 세 꼭짓점을 2D로 투영
    2. 면의 법선으로 광원과의 각도 계산 → 밝기 결정
    3. Bounding box 내 각 픽셀에 대해:
       - 무게중심 좌표로 삼각형 내부 판별
       - Z-buffer 비교 후 더 가까우면 문자 갱신
    """
    # 2D 투영
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

    # 면 법선 벡터 계산 → 라이팅
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-10:
        return  # 퇴화 삼각형 방지
    normal = normal / norm_len

    # 법선이 항상 바깥쪽(카메라 쪽)을 향하도록 보정.
    # 정이십면체는 원점 중심이므로, 면의 중심점과 법선의 내적이
    # 양수여야 바깥 방향. 음수면 뒤집어줌.
    center = (v0 + v1 + v2) / 3
    if np.dot(normal, center) < 0:
        normal = -normal

    # Lambert 반사 모델: 밝기 = max(0, 법선·광원)
    brightness = max(0, np.dot(normal, light_dir))
    char_idx = int(brightness * (len(shade_chars) - 1))
    char = shade_chars[char_idx]

    # 각 픽셀 래스터라이즈
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            w0, w1, w2 = barycentric(x, y, x0, y0, x1, y1, x2, y2)

            # 삼각형 내부인지 확인
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # 무게중심 좌표로 Z값 보간
                z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

                # Z-buffer 테스트: 더 가까운 면이면 갱신
                if z < z_buffer[y][x]:
                    z_buffer[y][x] = z
                    frame[y][x] = char


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────

# ANSI 이스케이프 시퀀스
# \033[H  : 커서를 (0,0)으로 이동 (화면을 지우지 않고 덮어쓰기)
# \033[2J : 화면 전체 지우기 (최초 1회만 사용)
# \033[?25l : 커서 숨기기
# \033[?25h : 커서 보이기

import sys

# 회전 각도 (라디안)
angle_x = 0.0
angle_y = 0.0
angle_z = 0.0

# 최초 화면 초기화 + 커서 숨기기
sys.stdout.write('\033[2J\033[?25l')
sys.stdout.flush()

try:
    while True:
        # 커서를 좌상단으로 이동 (화면을 지우지 않음 → 깜빡임 없음)
        sys.stdout.write('\033[H')

        # ── 1. 회전 행렬 합성 (Z → Y → X 순) ──
        R = rotation_matrix_z(angle_z) @ rotation_matrix_y(angle_y) @ rotation_matrix_x(angle_x)
        rotated = vertices @ R.T  # 모든 꼭짓점에 회전 적용

        # ── 2. 프레임 버퍼 & Z-버퍼 초기화 ──
        frame = [[' '] * WIDTH for _ in range(HEIGHT)]
        z_buffer = [[float('inf')] * WIDTH for _ in range(HEIGHT)]

        # ── 3. 면 채우기 (Z-buffer + flat shading) ──
        for i, j, k in faces:
            v0, v1, v2 = rotated[i], rotated[j], rotated[k]

            if is_face_visible(v0, v1, v2):
                fill_triangle(frame, z_buffer, v0, v1, v2)

        # ── 4. 와이어프레임 엣지 그리기 ──
        # 보이는 면의 엣지만 수집 (중복 제거)
        visible_edges = set()
        for i, j, k in faces:
            v0, v1, v2 = rotated[i], rotated[j], rotated[k]

            if is_face_visible(v0, v1, v2):
                # 엣지를 (작은 인덱스, 큰 인덱스)로 정규화하여 중복 방지
                visible_edges.add((min(i, j), max(i, j)))
                visible_edges.add((min(j, k), max(j, k)))
                visible_edges.add((min(k, i), max(k, i)))

        for i, j in visible_edges:
            x0, y0 = project(rotated[i])
            x1, y1 = project(rotated[j])

            if x0 is None or x1 is None:
                continue

            draw_line(frame, x0, y0, x1, y1, '*')

        # ── 5. 화면 출력 ──
        # 전체 프레임을 하나의 문자열로 만들어 한 번에 write (깜빡임 방지)
        output = '\n'.join(''.join(row) for row in frame)
        sys.stdout.write(output)
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