from .model_yolo import YOLOModel


def train_model(
    train_dir, 
    val_dir, 
    epochs=100, 
    model_size="s",
    img_size=640,
    batch_size=None,
    lr0=0.005,
    augment=True,
    mosaic=1.0,
    mixup=0.3
):
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
        model_size (str): 모델 크기 ('n', 's', 'm', 'l')
        img_size (int): 입력 이미지 크기
        batch_size (int): 배치 크기
        lr0 (float): 초기 학습률
        augment (bool): 데이터 증강 활성화 여부
        mosaic (float): 모자이크 증강 비율 (0.0-1.0)
        mixup (float): 믹스업 증강 비율 (0.0-1.0)

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

    model_name = f"yolov8{model_size}.pt"
    
    yolom = YOLOModel(model_name=model_name)
    results = yolom.train(
        train_dir,
        val_dir,
        epochs=epochs,
        class_names=class_names,
        img_size=img_size,
        batch_size=batch_size,
        lr0=lr0,
        augment=augment,
        mosaic=mosaic,
        mixup=mixup,
        lrf=0.01,  # 최종 학습률 비율
        degrees=15.0,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        box=7.5,       # 박스 손실 가중치
        cls=1.5,       # 클래스 손실 가중치
        dfl=1.5,       # 분포 초점 손실 가중치
    )

    return yolom
