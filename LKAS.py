
import time
import signal
import sys
import subprocess

import cv2
import numpy as np
import RPi.GPIO as GPIO

from picamera import PiCamera
from picamera.array import PiRGBArray
from nanpy import ArduinoApi, SerialManager

from hcsr04 import hcsr04
from motor import motorA, motorB
from servo import servo_motor

CAMERA_WIDTH = 400
CAMERA_HEIGHT = 300
CAMERA_FPS = 30

BUZZER_PIN = 4
OBSTACLE_DISTANCE = 20
CLOSE_DISTANCE = 10

APP_PATH = "/home/pi/Desktop/LKAS/app.py"

connection = SerialManager()
arduino = ArduinoApi(connection=connection)
arduino.pinMode(BUZZER_PIN, arduino.OUTPUT)

def gray(image):
    """컬러 영상을 그레이스케일 영상으로 변환한다."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def GaussianBlur(image):
    """가우시안 블러로 영상의 노이즈를 줄인다."""
    return cv2.GaussianBlur(image, (3, 3), 0)


def CannyEdge(image):
    """Canny 알고리즘으로 영상의 에지를 검출한다."""
    return cv2.Canny(image, 5, 300)


def roi(image, vertices):
    """관심 영역(ROI) 외부의 에지를 제거한다."""
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)

    return cv2.bitwise_and(image, mask)


def DetectRegion(image, vertices):
    """영상에 관심 영역의 경계선을 표시한다."""
    cv2.polylines(image, vertices, True, (255, 0, 0), 3)
    return image

def HoughLinesP(image):
    """확률적 허프 변환으로 에지 영상에서 직선을 검출한다."""
    return cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=5,
        minLineLength=1,
        maxLineGap=1
    )


def get_fitline(image, lines):
    """
    여러 허프 직선을 하나의 대표 직선으로 근사한다.
    반환값: [x1, y1, x2, y2]
    """
    points = lines.reshape(-1, 2)

    vx, vy, x, y = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    ).flatten()

    # 직선이 거의 수평이면 계산을 수행하지 않는다.
    if abs(vy) < 1e-6:
        return None

    y1 = image.shape[0] - 1
    y2 = int(image.shape[0] / 2 + 100)

    x1 = int((y1 - y) / vy * vx + x)
    x2 = int((y2 - y) / vy * vx + x)

    return [x1, y1, x2, y2]


def draw_fit_line(image, line):
    """계산된 대표 직선을 영상에 표시한다."""
    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    return line


def split_lane_lines(lines):
    """
    허프 직선의 기울기를 기준으로 왼쪽 차선과
    오른쪽 차선 후보를 분류한다.
    """
    lines = np.asarray(lines).reshape(-1, 4)

    slope_degree = (
        np.arctan2(
            lines[:, 1] - lines[:, 3],
            lines[:, 0] - lines[:, 2]
        ) * 180 / np.pi
    )

    # 수평선에 가까운 직선을 제외한다.
    valid = (
        (np.abs(slope_degree) < 160) &
        (np.abs(slope_degree) > 95)
    )

    lines = lines[valid]
    slope_degree = slope_degree[valid]

    left_lines = lines[slope_degree > 0]
    right_lines = lines[slope_degree < 0]

    return left_lines, right_lines

def find_vanishing_point(left_line, right_line):
    """
    좌우 대표 직선의 교점을 계산한다.
    교점의 (x, y)를 반환하며, 계산할 수 없으면 None을 반환한다.
    """
    if left_line is None or right_line is None:
        return None

    x1, y1, x2, y2 = left_line
    x3, y3, x4, y4 = right_line

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )

    # 두 직선이 평행하거나 거의 평행한 경우
    if abs(denominator) < 1e-6:
        return None

    determinant1 = x1 * y2 - y1 * x2
    determinant2 = x3 * y4 - y3 * x4

    x_intersect = (
        determinant1 * (x3 - x4)
        - (x1 - x2) * determinant2
    ) / denominator

    y_intersect = (
        determinant1 * (y3 - y4)
        - (y1 - y2) * determinant2
    ) / denominator

    return int(x_intersect), int(y_intersect)


def servoMotor(x_intersect_pixel, image_width):
    """소실점의 가로 위치를 서보모터 제어값으로 변환한다."""
    percentage = x_intersect_pixel / image_width * 100
    pwm_value = 45 + percentage * 0.9

    servo_motor(pwm_value)

def buzz(on_time, off_time):
    """부저를 지정한 시간 동안 켜고 끈다."""
    arduino.digitalWrite(BUZZER_PIN, arduino.HIGH)
    time.sleep(on_time)

    arduino.digitalWrite(BUZZER_PIN, arduino.LOW)
    time.sleep(off_time)


def ultrasonic_sensor():
    """장애물 거리 측정 및 모터·부저 제어."""
    distance = hcsr04()

    # 가까운 장애물을 먼저 검사해야 한다.
    if distance <= CLOSE_DISTANCE:
        motorA(0)
        motorB(0)
        buzz(1, 0)

    elif distance <= OBSTACLE_DISTANCE:
        motorA(0)
        motorB(0)
        buzz(1, 1)

    else:
        motorA(255)
        motorB(255)

def process_lane_detection(frame):
    """
    한 프레임에서 차선을 검출하고 소실점을 계산한다.
    처리 결과 영상과 소실점을 반환한다.
    """
    image = frame.copy()
    height, width = frame.shape[:2]

    vertices = np.array([[
        (0, height),
        (width * 1 // 8, height * 5 // 7),
        (width * 7 // 8, height * 5 // 7),
        (width, height)
    ]], dtype=np.int32)

    # 영상 전처리
    gray_image = gray(frame)
    blurred_image = GaussianBlur(gray_image)
    edges = CannyEdge(blurred_image)
    roi_image = roi(edges, vertices)

    # 결과 영상에 ROI 경계 표시
    DetectRegion(image, vertices)

    # 허프 직선 검출
    detected_lines = HoughLinesP(roi_image)

    if detected_lines is None:
        return image, None

    # 좌우 차선 후보 분류
    left_lines, right_lines = split_lane_lines(detected_lines)

    if len(left_lines) == 0 or len(right_lines) == 0:
        return image, None

    # 좌우 대표 직선 계산
    left_line = get_fitline(roi_image, left_lines)
    right_line = get_fitline(roi_image, right_lines)

    # 대표 직선 표시
    draw_fit_line(image, left_line)
    draw_fit_line(image, right_line)

    # 소실점 계산
    vanishing_point = find_vanishing_point(left_line, right_line)

    if vanishing_point is not None:
        x, y = vanishing_point
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

    return image, vanishing_point

def shutdown(camera=None):
    """모터와 부저를 정지하고 하드웨어 및 창을 정리한다."""
    motorA(0)
    motorB(0)

    arduino.digitalWrite(BUZZER_PIN, arduino.LOW)

    if camera is not None:
        camera.close()

    GPIO.cleanup()
    cv2.destroyAllWindows()


def signal_handler(signum, frame):
    """Ctrl+C 입력 시 프로그램을 종료한다."""
    print("\n프로그램을 종료합니다.")
    shutdown()
    sys.exit(0)

def main():
    camera = None

    # 보조 애플리케이션 실행
    subprocess.Popen(["python3", APP_PATH])

    try:
        # 초기 모터 속도 설정
        motorA(250)
        motorB(250)

        # 카메라 초기화
        camera = PiCamera()
        camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
        camera.framerate = CAMERA_FPS

        raw_capture = PiRGBArray(
            camera,
            size=(CAMERA_WIDTH, CAMERA_HEIGHT)
        )

        time.sleep(0.1)

        # 프레임 단위 영상 처리
        for frame in camera.capture_continuous(
            raw_capture,
            format="bgr",
            use_video_port=True
        ):
            image = frame.array

            # 차선 검출 및 소실점 계산
            result_image, vanishing_point = process_lane_detection(image)

            # 소실점 위치에 따른 조향 제어
            if vanishing_point is not None:
                x_intersect_pixel, _ = vanishing_point
                servoMotor(x_intersect_pixel, CAMERA_WIDTH)

            # 초음파 장애물 감지
            ultrasonic_sensor()

            # 결과 영상 출력
            cv2.imshow("Lane Detection", result_image)

            key = cv2.waitKey(1) & 0xFF

            # 다음 프레임을 위해 버퍼 초기화
            raw_capture.truncate(0)

            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.")

    finally:
        shutdown(camera)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()

