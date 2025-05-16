import os
import cv2
import random
import shutil
import time

from ultralytics import YOLO


class YOLOModel:
    """
    YOLOv8(You Only Look Once) 모델을 위한 래퍼 클래스입니다.

    이 클래스는 Ultralytics 라이브러리의 YOLO 구현체를 활용하여 다음 기능을 제공합니다:
    1. 사전 훈련된 YOLO 모델 로드 또는 커스텀 데이터셋으로 학습
    2. 이미지에서 객체 감지 수행
    3. 학습된 모델 저장 및 로드

    YOLO는 단일 스테이지 객체 감지 알고리즘으로, 빠른 추론 속도와 높은 정확도를 제공합니다.
    """

    def __init__(self, model_name="yolov8n.pt", device="auto"):
        """
        YOLOv8 모델을 초기화합니다.

        매개변수:
            model_name (str): 사용할 모델의 이름 또는 경로
                - "yolov8n.pt": 나노 모델 (가장 작고 빠름, 기본값)
                - "yolov8s.pt": 소형 모델 (속도와 정확도 균형)
                - "yolov8m.pt": 중형 모델 (중간 성능)
                - "yolov8l.pt": 대형 모델 (높은 정확도)
                - "yolov8x.pt": 초대형 모델 (최고 정확도)
                - 커스텀 학습된 모델의 경로 (예: "runs/detect/exp/weights/best.pt")
            device (str): 모델 실행 장치
                - "auto": 자동 선택 (CUDA 사용 가능시 GPU, 아니면 CPU)
                - "cpu": CPU만 사용
                - "cuda": GPU 사용 (CUDA 지원 필요)
                - "0" 또는 "1": 특정 GPU 장치 지정

        참고:
            - 초기화 시점에는 실제 모델이 메모리에 로드되지 않으며, 첫 predict() 호출 시 로드됩니다.
            - 큰 모델일수록 더 정확하지만 속도가 느리고 메모리 사용량이 증가합니다.
        """
        self.model_name = model_name  # 모델 가중치 파일 이름 또는 경로
        self.device = device  # 실행 장치 (CPU/GPU)
        self.best_weight = None  # 학습 후 생성된 최적 가중치 경로
        self.model = None  # 실제 YOLO 모델 객체 (지연 로딩)

    def _create_data_yaml(self, train_dir, val_dir, class_names):
        """
        YOLO 학습에 필요한 data.yaml 파일을 생성합니다.

        매개변수:
            train_dir (str): 학습 데이터 디렉토리 경로
            val_dir (str): 검증 데이터 디렉토리 경로
            class_names (list): 클래스 이름 목록

        반환값:
            str: data.yaml 파일 경로
        """
        import yaml

        # 상위 디렉토리 결정
        parent_dir = os.path.dirname(train_dir) if os.path.dirname(train_dir) else "."
        yaml_path = os.path.join(parent_dir, "data.yaml")

        # 절대 경로로 변환
        train_dir_abs = os.path.abspath(train_dir)
        val_dir_abs = os.path.abspath(val_dir) if val_dir else train_dir_abs

        # 클래스 이름이 없으면 기본값 사용
        if not class_names:
            class_names = ["object"]

        # YAML 데이터 구성
        data = {
            "train": train_dir_abs,
            "val": val_dir_abs,
            "names": {i: name for i, name in enumerate(class_names)},
            "nc": len(class_names),
        }

        # YAML 파일 작성
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"[YOLOModel] data.yaml 파일 생성됨: {yaml_path}")
        print(f"[YOLOModel] YAML 내용:\n{data}")

        return yaml_path

    def train(
        self,
        train_dir,
        val_dir=None,
        epochs=50,
        device=None,
        class_names=None,
        force_retrain=True,
        img_size=640,
        batch_size=None,
        project="runs/detect",
        name="train",
        lr0=0.01,
        lrf=0.01,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        box=7.5,
        cls=0.5,
        dfl=1.5
    ):
        """
        YOLO 모델을 커스텀 클래스로 학습합니다.

        매개변수:
            train_dir (str): 학습 데이터 디렉토리 경로
            val_dir (str): 검증 데이터 디렉토리 경로
            epochs (int): 학습 에포크 수
            device (str): 학습 장치 ('cpu' 또는 'cuda')
            class_names (list): 사용자 정의 클래스 이름 목록
            force_retrain (bool): 이미 학습된 모델이 있어도 재학습 실행 여부
            img_size (int): 이미지 크기
            batch_size (int): 배치 크기
            project (str): 프로젝트 경로
            name (str): 학습 이름
            lr0 (float): 초기 학습률
            lrf (float): 최종 학습률 비율 
            augment (bool): 데이터 증강 사용 여부
            mosaic (float): 모자이크 증강 비율 (0.0-1.0)
            mixup (float): 믹스업 증강 비율 (0.0-1.0)
            degrees (float): 회전 증강 각도
            scale (float): 크기 조정 비율
            flipud (float): 상하 반전 확률
            fliplr (float): 좌우 반전 확률
            box (float): 박스 손실 가중치
            cls (float): 클래스 손실 가중치
            dfl (float): 분포 초점 손실 가중치
        """
        try:
            import torch
            from ultralytics import YOLO

            # 이미 학습된 모델 확인
            best_weight_path = os.path.join(project, name, "weights", "best.pt")
            if os.path.exists(best_weight_path) and not force_retrain:
                print(
                    f"[YOLOModel] 이미 학습된 모델이 발견되었습니다: {best_weight_path}"
                )
                print(f"[YOLOModel] 학습을 건너뛰고 기존 모델을 사용합니다.")
                print(f"[YOLOModel] 재학습이 필요하면 force_retrain=True로 설정하세요.")
                self.model = YOLO(best_weight_path)
                self.model_name = best_weight_path
                return best_weight_path

            # 장치 설정
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[YOLOModel] YOLO 모델 학습 시작")
            print(f"[YOLOModel] 장치: {device}")

            # 커스텀 클래스 이름 설정
            if class_names:
                self.target_classes = class_names
                print(f"[YOLOModel] 학습할 커스텀 클래스: {class_names}")
            else:
                # 기본 클래스 이름 (최소한 하나는 필요)
                self.target_classes = ["object"]
                print(
                    "[YOLOModel] 경고: 클래스 이름이 제공되지 않아 기본값 'object' 사용"
                )

            # data.yaml 파일 생성
            yaml_path = self._create_data_yaml(train_dir, val_dir, self.target_classes)

            # YOLO 모델 로드 (기존 가중치에서 시작)
            self.model = YOLO(self.model_name)

            # 배치 크기 설정
            if batch_size is None:
                batch_size = 8 if device == "cuda" else 4

            # 학습 실행 - 모든 하이퍼파라미터 전달
            results = self.model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                verbose=True,
                project=project,
                name=name,
                lr0=lr0,
                lrf=lrf,
                augment=augment,
                mosaic=mosaic,
                mixup=mixup,
                degrees=degrees,
                scale=scale,
                flipud=flipud,
                fliplr=fliplr,
                box=box,
                cls=cls,
                dfl=dfl
            )

            # 학습된 모델 저장 경로
            if os.path.exists(best_weight_path):
                print(f"[YOLOModel] 학습 완료: 최적 가중치 저장됨 ({best_weight_path})")
                self.model = YOLO(best_weight_path)
                self.model_name = best_weight_path

            return self.model_name

        except Exception as e:
            print(f"[YOLOModel] 학습 중 오류 발생: {str(e)}")
            return None

    def predict(self, img_bgr, conf_thresh=0.25):
        """
        이미지에서 객체를 감지합니다.

        매개변수:
            img_bgr (ndarray): OpenCV BGR 형식의 입력 이미지
            conf_thresh (float): 객체 감지 신뢰도 임계값 (0.0 ~ 1.0)

        반환값:
            list: 감지된 객체의 바운딩 박스 목록
                - 각 항목은 [x1, y1, x2, y2, score, class_id] 형식
                - class_id: 객체 클래스 ID (0-16)
        """
        # 필요시 모델 로드 (지연 로딩)
        if self.model is None:
            if self.best_weight and os.path.exists(self.best_weight):
                # 학습된 가중치 사용
                print(f"[YOLOModel.predict] 학습된 가중치 로드: {self.best_weight}")
                self.model = YOLO(self.best_weight)
            else:
                # 사전 훈련된 모델 사용
                print(f"[YOLOModel.predict] 사전 훈련된 모델 로드: {self.model_name}")
                self.model = YOLO(self.model_name)

        # 객체 감지 수행
        results = self.model.predict(img_bgr, conf=conf_thresh)
        result = results[0]  # 첫 번째 이미지 결과

        # 결과를 표준 형식으로 변환 [x1, y1, x2, y2, score, class_id]
        boxes = []
        if hasattr(result, "boxes") and result.boxes is not None:
            # 감지된 객체가 있는 경우
            for box in result.boxes.data:
                box_data = box.cpu().numpy()
                # x1, y1, x2, y2, confidence, class_id
                x1, y1, x2, y2 = map(float, box_data[:4])
                conf = float(box_data[4])
                class_id = int(box_data[5]) if len(box_data) > 5 else 0

                boxes.append([x1, y1, x2, y2, conf, class_id])

        return boxes

    def save(self, path):
        """
        현재 로드된 YOLO 모델을 파일로 저장합니다.

        매개변수:
            path (str): 모델을 저장할 파일 경로
                - 일반적으로 .pt 확장자 사용
                - 경로에 존재하지 않는 디렉토리가 있으면 자동 생성

        예외:
            RuntimeError: 모델이 로드되지 않은 상태에서 호출 시 발생

        참고:
            - 모델은 가중치, 구조, 하이퍼파라미터를 포함하여 저장됨
            - 저장된 모델은 load() 메서드로 다시 로드 가능
        """
        # 모델 존재 여부 확인
        if self.model is None:
            raise RuntimeError(
                "[YOLOModel.save] 저장할 모델이 없습니다. 먼저 학습하거나 모델을 로드하세요."
            )

        # 파일 확장자 확인 및 추가
        if not path.endswith(".pt"):
            path += ".pt"

        # 상위 디렉토리 생성 (없는 경우)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # 모델 저장
        self.model.save(path)
        print(f"[YOLOModel.save] 모델 저장 완료: {path}")

    def load(self, path):
        """
        저장된 YOLO 모델을 파일에서 로드합니다.

        매개변수:
            path (str): 로드할 모델 파일 경로
                - .pt 확장자의 파일이어야 함
                - 사전 훈련된 모델명(예: "yolov8n.pt") 또는 경로 가능

        예외:
            FileNotFoundError: 지정된 경로에 모델 파일이 없을 경우 발생

        참고:
            - 이전에 로드된 모델이 있다면 메모리에서 해제하고 새 모델 로드
            - 학습/추론에 사용할 모델을 명시적으로 지정할 때 유용
        """
        # 파일 존재 여부 확인
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[YOLOModel.load] 모델 파일을 찾을 수 없습니다: {path}"
            )

        # 기존 모델 해제 및 새 모델 로드
        if self.model is not None:
            del self.model  # 기존 모델 메모리 해제

        self.model = YOLO(path)
        self.best_weight = path
        print(f"[YOLOModel.load] 모델 로드 완료: {path}")


def create_subset(source_dir, target_dir, sampling_ratio=0.3, min_samples=10):
    """주요 클래스는 30% 샘플링, 소수 클래스는 전부 유지하는 전략"""
    os.makedirs(target_dir, exist_ok=True)
    label_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
    
    print(f"\n[데이터 축소] 원본 파일 수: {len(label_files)}개")
    print(f"[데이터 축소] 소스 경로: {source_dir}")
    print(f"[데이터 축소] 타겟 경로: {target_dir}")
    print(f"[데이터 축소] 샘플링 비율: {sampling_ratio*100:.1f}% (소수 클래스 100% 유지)")
    print("-" * 60)
    
    # 클래스별 파일 분류
    print("[데이터 축소] 파일 분석 중...")
    class_files = {}
    for i, file in enumerate(label_files):
        if i % 1000 == 0 and i > 0:
            print(f"  - {i}/{len(label_files)} 파일 처리 중... ({i/len(label_files)*100:.1f}%)")
            
        try:
            with open(os.path.join(source_dir, file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id not in class_files:
                            class_files[class_id] = []
                        class_files[class_id].append(file)
                        break  # 첫 객체의 클래스만 확인
        except Exception as e:
            print(f"  - 경고: 파일 {file} 처리 중 오류: {str(e)}")
    
    # 클래스별 샘플링
    selected_files = set()
    print("\n[데이터 축소] 클래스별 샘플링:")
    print(f"{'클래스ID':<10}{'원본 수':<10}{'샘플 수':<10}{'비율':<10}")
    print("-" * 40)
    
    total_orig = 0
    total_sampled = 0
    
    for class_id, files in sorted(class_files.items()):
        total_orig += len(files)
        # 소수 클래스(경찰차 등)는 전부 포함
        if len(files) < 150:  # 소수 클래스 기준
            sample_size = len(files)
            rare_class = True
        else:
            sample_size = max(min_samples, int(len(files) * sampling_ratio))
            rare_class = False
        
        if len(files) <= sample_size:
            samples = files
        else:
            samples = random.sample(files, sample_size)
        
        selected_files.update(samples)
        total_sampled += len(samples)
        
        ratio = len(samples) / len(files) * 100 if len(files) > 0 else 0
        status = "전체유지" if rare_class else f"{ratio:.1f}%"
        print(f"{class_id:<10}{len(files):<10}{len(samples):<10}{status:<10}")
    
    # 결과 요약
    print("-" * 40)
    print(f"총계: {total_orig} -> {total_sampled} ({total_sampled/total_orig*100:.1f}%)")
    
    # 파일 복사
    print(f"\n[데이터 축소] {len(selected_files)}개 파일 복사 중...")
    start_time = time.time()
    
    for i, file in enumerate(selected_files):
        if (i+1) % 100 == 0 or (i+1) == len(selected_files):
            elapsed = time.time() - start_time
            files_per_sec = (i+1) / elapsed if elapsed > 0 else 0
            eta = (len(selected_files) - (i+1)) / files_per_sec if files_per_sec > 0 else 0
            progress = (i+1) / len(selected_files) * 100
            
            print(f"  - 진행: {progress:.1f}% ({i+1}/{len(selected_files)}) - "
                  f"속도: {files_per_sec:.1f} 파일/초, "
                  f"남은 시간: {eta:.1f}초")
        
        shutil.copy2(os.path.join(source_dir, file), target_dir)
    
    total_time = time.time() - start_time
    print(f"\n[데이터 축소] 완료: {len(selected_files)}개 파일 복사됨 (소요시간: {total_time:.2f}초)")
    return len(selected_files)
