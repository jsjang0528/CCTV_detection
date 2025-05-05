from .model_yolo import YOLOModel


def train_model(train_dir, val_dir, epochs=5):
    """
    지정된 유형의 객체 감지 모델을 학습합니다.

    매개변수:
        model_type (str): 학습할 모델 유형
            - 'traditional': HOG+SVM 모델 (전통적인 컴퓨터 비전 접근법)
            - 'yolo': YOLOv8 모델 (딥러닝 기반 접근법)
        train_dir (str): 학습 데이터 디렉토리 경로
            - 구조: train_dir/images, train_dir/labels
        val_dir (str): 검증 데이터 디렉토리 경로 (YOLO 모델에서만 사용)
        epochs (int): 학습 에포크 수 (YOLO 모델에서만 사용)

    반환값:
        모델 객체 YOLOModel

    주의:
        - 'yolo' 모델은 Ultralytics YOLO 라이브러리를 사용하여 학습 수행
    """

    class_names = [
        "경차/세단",
        "SUV/승합차",
        "트럭",
        "버스(소형, 대형)",
        "통학버스(소형,대형)",
        "경찰차",
        "구급차",
        "소방차",
        "견인차",
        "기타 특장차",
        "성인",
        "어린이",
        "오토바이",
        "자전거 / 기타 전동 이동체",
        "라바콘",
        "삼각대",
        "기타",
    ]

    yolom = YOLOModel(model_name="yolov8n.pt")  # 기본 모델: YOLOv8 nano
    yolom.train(
        train_dir,
        val_dir,
        epochs=epochs,
        class_names=class_names,
    )

    return yolom
