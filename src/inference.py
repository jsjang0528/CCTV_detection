import os

import cv2


def preprocess_image(img, use_gray=False, use_clahe=True):
    """
    객체 감지 추론을 위한 이미지 전처리 함수

    매개변수:
        img (numpy.ndarray): 입력 이미지 (BGR 형식)
        use_gray (bool): 그레이스케일 변환 적용 여부
        use_clahe (bool): CLAHE 대비 향상 적용 여부

    반환값:
        numpy.ndarray: 전처리된 이미지 (BGR 형식)
    """
    if img is None:
        raise ValueError("[preprocess_image] 이미지가 비어 있습니다")

    # 이미지 복사본 생성 (원본 변경 방지)
    processed_img = img.copy()

    # 그레이스케일 변환 (선택 사항)
    if use_gray:
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 3채널 형식 유지

    # CLAHE 적용 (선택 사항)
    if use_clahe:
        # YCrCb 색상 공간으로 변환 (Y: 밝기 채널)
        ycrcb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        # 밝기 채널에만 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        # 채널 합치기
        merged = cv2.merge((y, cr, cb))
        processed_img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    return processed_img


def inference_yolo(yolo_model, test_dir, out_label_dir=None):
    """
    YOLOv8 모델을 사용하여 테스트 이미지에서 객체 감지 추론을 수행합니다.

    클래스 ID:
    0: 경차/세단       1: SUV/승합차        2: 트럭
    3: 버스(소형, 대형) 4: 통학버스(소형,대형) 5: 경찰차
    6: 구급차         7: 소방차            8: 견인차
    9: 기타 특장차     10: 성인            11: 어린이
    12: 오토바이       13: 자전거/기타 전동 이동체  14: 라바콘
    15: 삼각대        16: 기타

    매개변수:
        yolo_model: YOLOv8 모델 인스턴스
        test_dir: 테스트 이미지 디렉토리 경로
        out_label_dir: 출력 라벨 디렉토리 경로 (기본값: test_dir/labels_pred)

    반환값:
        출력 라벨 디렉토리 경로
    """
    if out_label_dir is None:
        out_label_dir = os.path.join(test_dir, "labels_pred")

    os.makedirs(out_label_dir, exist_ok=True)
    img_dir = os.path.join(test_dir, "images")
    files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    )

    print(f"[inference_yolo] 테스트 이미지: {len(files)}개")

    for f in files:
        # 이미지 경로 및 출력 파일 경로 생성
        imgp = os.path.join(img_dir, f)
        base = os.path.splitext(f)[0]
        out_txt = os.path.join(out_label_dir, base + ".txt")

        # 이미지 로드
        img = cv2.imread(imgp)
        if img is None:
            print(f"[경고] 이미지를 읽을 수 없습니다: {imgp}")
            continue

        # 이미지 전처리 적용 (그레이스케일과 CLAHE만)
        # 파라미터 값은 필요에 따라 조정 가능
        processed_img = preprocess_image(img, use_gray=False, use_clahe=True)

        # 이미지 크기 (YOLO 좌표 정규화에 사용)
        h, w, _ = processed_img.shape

        # YOLO 모델로 객체 감지 (전처리된 이미지 사용)
        results = yolo_model.predict(processed_img, conf_thresh=0.25)

        # 결과를 YOLO 형식으로 변환
        lines = []
        for box in results:
            # 박스 정보 추출 (x1, y1, x2, y2, 신뢰도, 클래스 ID)
            if len(box) >= 6:  # 클래스 ID가 포함된 경우
                x1, y1, x2, y2, sc, cls_id = box[:6]
            else:  # 클래스 ID가 없는 경우 (기본값: 0)
                x1, y1, x2, y2, sc = box[:5]
                cls_id = 0  # 기본값: 경차/세단

            # 픽셀 좌표를 YOLO 형식(중심점, 너비, 높이, 정규화)으로 변환
            bw = x2 - x1  # 박스 너비
            bh = y2 - y1  # 박스 높이
            x_ctr = x1 + bw / 2  # 중심 X
            y_ctr = y1 + bh / 2  # 중심 Y

            # 이미지 크기로 정규화 (0~1 범위)
            x_ctr /= w
            y_ctr /= h
            bw /= w
            bh /= h

            # YOLO 형식 라인 생성: "class_id x_center y_center width height score"
            line = f"{int(cls_id)} {x_ctr:.6f} {y_ctr:.6f} {bw:.6f} {bh:.6f} {sc:.3f}"
            lines.append(line)

        # 결과를 텍스트 파일로 저장
        with open(out_txt, "w") as fw:
            for ln in lines:
                fw.write(ln + "\n")

    print(
        f"[inference_yolo] 추론 완료: {len(files)}개 이미지, 결과 저장 경로: {out_label_dir}"
    )
    return out_label_dir


def generate_prediction(model, test_dir, out_label_dir=None):
    """
    지정된 모델을 사용하여 테스트 이미지에서 객체를 감지하고 결과를 YOLO 형식으로 저장합니다.

    매개변수:
        model_type (str): 모델 유형 ('traditional' 또는 'yolo')
        model: 학습된 모델 객체 (HogSVMModel 또는 YOLOModel)
        test_dir (str): 테스트 이미지가 있는 디렉토리 경로
        out_label_dir (str, optional): 결과 라벨을 저장할 디렉토리 경로

    반환값:
        str: 결과 라벨 디렉토리 경로
    """
    # YOLO 모델 추론
    lbl_dir = inference_yolo(model, test_dir)
    return lbl_dir
