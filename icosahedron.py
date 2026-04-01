import numpy as np
import time
import os

WIDTH = 100
HEIGHT = 50

phi = (1 + np.sqrt(5)) / 2

vertices = np.array([
    [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
], dtype=float) * 0.8

edges = [
    # 상단 오각형 (꼭짓점 4 중심)
    (4, 0), (4, 6), (4, 1), (4, 9), (4, 8),
    
    # 하단 오각형 (꼭짓점 5 중심)
    (5, 2), (5, 7), (5, 3), (5, 10), (5, 11),
    
    # 상단 오각형 모서리
    (0, 8), (8, 9), (9, 1), (1, 6), (6, 0),
    
    # 하단 오각형 모서리
    (2, 10), (10, 11), (11, 3), (3, 7), (7, 2),
    
    # 지그재그 연결 (상단-하단)
    (0, 2), (8, 2), (8, 11), (9, 11), (9, 3),
    (1, 3), (1, 7), (6, 7), (6, 10), (4, 10)
]

def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def project(point):
    z = point[2] + 12 # 카메라 거리조절
    if z <= 0:
        return None, None
    
    FOV = 200
    x = int(point[0] * 256 / z + WIDTH / 2)
    y = int(-point[1] * 256 / z * 0.5 + HEIGHT / 2)
    
    return x, y

def draw_line(frame, x0, y0, x1, y1, char='*'):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if (0 <= x0 < WIDTH) and (0 <= y0 < HEIGHT):
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
        

angle_x = angle_y = angle_z = 0

# main
while True:
    os.system('cls')
    
    # 1. 회전 먼저 계산
    R = rotation_matrix_z(angle_z) @ rotation_matrix_y(angle_y) @ rotation_matrix_x(angle_x)
    rotated = vertices @ R.T
    
    # 2. 프레임 버퍼 초기화
    frame = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]
    
    # 3. 회전된 점들을 그리기
    for v in rotated:
        x, y = project(v)
        if x is not None and 0 <= x < WIDTH and 0 <= y < HEIGHT:
            frame[y][x] = '*'
    
    for i, j in edges:
        x0, y0 = project(rotated[i])
        x1, y1 = project(rotated[j])
    
        if x0 is not None and x1 is not None:
            draw_line(frame, x0, y0, x1, y1, '*')
    # 4. 출력
    for row in frame:
        print(''.join(row))
    
    # 5. 각도 업데이트
    angle_x += 0.03
    angle_y += 0.02
    angle_z += 0.015
    
    time.sleep(0.05)