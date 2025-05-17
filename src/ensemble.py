import os
import glob
import numpy as np
import cv2
from ultralytics import YOLO
import torch

from .model_yolo import YOLOModel


def non_max_suppression(boxes, scores, labels, iou_threshold=0.5, img_width=640, img_height=480):
    """
    비최대 억제(Non-Maximum Suppression) 알고리즘으로 중복 박스 제거
    
    매개변수:
        boxes: 박스 좌표 배열 [x1, y1, x2, y2]
        scores: 각 박스의 신뢰도 점수
        labels: 각 박스의 클래스 라벨
        iou_threshold: IoU 임계값 (기본값: 0.5)
        img_width: 이미지 너비 (정규화된 좌표 변환용, 기본값: 640)
        img_height: 이미지 높이 (정규화된 좌표 변환용, 기본값: 480)
        
    반환값:
        선택된 박스, 점수, 라벨 인덱스
    """
    if len(boxes) == 0:
        return [], [], []
    
    # 픽셀 좌표로 변환 (예측이 정규화된 좌표인 경우)
    if np.max(boxes) <= 1.0:
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= img_width
        boxes[:, [1, 3]] *= img_height
    
    # 좌표 형식이 [cx, cy, w, h]인 경우 [x1, y1, x2, y2]로 변환
    if boxes.shape[1] == 4:
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # 신뢰도 점수로 인덱스 정렬 (내림차순)
    indices = np.argsort(scores)[::-1]
    
    keep = []  # 유지할 박스 인덱스
    
    while indices.size > 0:
        # 가장 높은 점수의 박스 선택
        i = indices[0]
        keep.append(i)
        
        # 마지막 박스면 종료
        if indices.size == 1:
            break
            
        # 나머지 박스와 IoU 계산
        box_i = boxes[i]
        rest_boxes = boxes[indices[1:]]
        
        # IoU 계산
        xx1 = np.maximum(box_i[0], rest_boxes[:, 0])
        yy1 = np.maximum(box_i[1], rest_boxes[:, 1])
        xx2 = np.minimum(box_i[2], rest_boxes[:, 2])
        yy2 = np.minimum(box_i[3], rest_boxes[:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        box_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        rest_area = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        union = box_area + rest_area - intersection
        
        iou = intersection / union
        
        # IoU 임계값보다 작은 박스만 유지
        inds = np.where(iou <= iou_threshold)[0]
        indices = indices[inds + 1]  # +1은 현재 박스를 건너뛰기 위함
    
    return boxes[keep], scores[keep], labels[keep]


def ensemble_predict(models, img, conf_thresh=0.25, iou_thresh=0.5):
    """
    여러 YOLO 모델의 예측 결과를 앙상블하여 최종 예측 생성
    
    매개변수:
        models: 모델 객체 리스트 (YOLOModel 인스턴스)
        img: 입력 이미지 (OpenCV BGR 형식)
        conf_thresh: 신뢰도 임계값 (기본값: 0.25)
        iou_thresh: NMS IoU 임계값 (기본값: 0.5)
        
    반환값:
        앙상블된 예측 결과 (NMS 적용 후)
    """
    all_predictions = []
    
    # 각 모델의 예측 수집
    for i, model in enumerate(models):
        try:
            predictions = model.predict(img, conf_thresh=conf_thresh)
            print(f"모델 {i+1}: {len(predictions)}개 객체 감지")
            all_predictions.extend(predictions)
        except Exception as e:
            print(f"모델 {i+1} 예측 오류: {str(e)}")
    
    if not all_predictions:
        print("앙상블 예측 결과가 없습니다.")
        return []
    
    # 이미지 크기 가져오기
    h, w = img.shape[:2]
    
    # 결과 형식 변환 [x1, y1, x2, y2, confidence, class_id]
    boxes = np.array([p[:4] for p in all_predictions])
    scores = np.array([p[4] for p in all_predictions])
    class_ids = np.array([p[5] for p in all_predictions])
    
    # NMS 적용
    final_boxes, final_scores, final_class_ids = non_max_suppression(
        boxes, scores, class_ids, iou_threshold=iou_thresh,
        img_width=w, img_height=h
    )
    
    # 최종 결과 형식으로 변환 [x1, y1, x2, y2, confidence, class_id]
    final_predictions = []
    for i in range(len(final_boxes)):
        final_predictions.append([
            final_boxes[i][0], final_boxes[i][1], 
            final_boxes[i][2], final_boxes[i][3], 
            final_scores[i], final_class_ids[i]
        ])
    
    print(f"앙상블 결과: {len(final_predictions)}개 객체 감지")
    return final_predictions


def test_time_augmentation(model, img, augmentations=['original', 'flip_h', 'flip_v', 'rotate90'], conf_thresh=0.25, iou_thresh=0.45):
    """
    테스트 시간 증강(TTA)을 수행하여 예측 정확도 향상
    
    매개변수:
        model: YOLOModel 인스턴스
        img: 입력 이미지 (OpenCV BGR 형식)
        augmentations: 적용할 증강 기법 목록
        conf_thresh: 신뢰도 임계값
        iou_thresh: NMS IoU 임계값
        
    반환값:
        TTA 적용 후 최종 예측 결과
    """
    h, w = img.shape[:2]
    all_predictions = []
    
    # 각 증강 기법에 대한 예측 수행
    for aug in augmentations:
        aug_img = img.copy()
        
        # 증강 적용
        if aug == 'flip_h':
            aug_img = cv2.flip(aug_img, 1)  # 수평 반전
        elif aug == 'flip_v':
            aug_img = cv2.flip(aug_img, 0)  # 수직 반전
        elif aug == 'rotate90':
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)  # 90도 회전
        elif aug == 'rotate180':
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_180)  # 180도 회전
        elif aug == 'rotate270':
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 270도 회전
        elif aug == 'blur':
            aug_img = cv2.GaussianBlur(aug_img, (5, 5), 0)  # 가우시안 블러
        
        # 예측 수행
        predictions = model.predict(aug_img, conf_thresh=conf_thresh)
        
        # 증강 기법에 따른 좌표 역변환
        for i, pred in enumerate(predictions):
            x1, y1, x2, y2, conf, class_id = pred
            
            # 증강에 따른 좌표 변환
            if aug == 'flip_h':
                # x 좌표 반전
                x1_new = w - x2
                x2_new = w - x1
                predictions[i][0] = x1_new
                predictions[i][2] = x2_new
            
            elif aug == 'flip_v':
                # y 좌표 반전
                y1_new = h - y2
                y2_new = h - y1
                predictions[i][1] = y1_new
                predictions[i][3] = y2_new
            
            elif aug == 'rotate90':
                # 90도 회전 역변환
                # (x,y) -> (y, w-x)
                x1_new, y1_new = y1, w - x2
                x2_new, y2_new = y2, w - x1
                predictions[i][0] = min(x1_new, x2_new)
                predictions[i][1] = min(y1_new, y2_new)
                predictions[i][2] = max(x1_new, x2_new)
                predictions[i][3] = max(y1_new, y2_new)
            
            elif aug == 'rotate180':
                # 180도 회전 역변환
                x1_new, y1_new = w - x2, h - y2
                x2_new, y2_new = w - x1, h - y1
                predictions[i][0] = x1_new
                predictions[i][1] = y1_new
                predictions[i][2] = x2_new
                predictions[i][3] = y2_new
                
            elif aug == 'rotate270':
                # 270도 회전 역변환
                # (x,y) -> (h-y, x)
                x1_new, y1_new = h - y2, x1
                x2_new, y2_new = h - y1, x2
                predictions[i][0] = min(x1_new, x2_new)
                predictions[i][1] = min(y1_new, y2_new)
                predictions[i][2] = max(x1_new, x2_new)
                predictions[i][3] = max(y1_new, y2_new)
        
        # 결과 추가
        all_predictions.extend(predictions)
    
    # 결과가 없으면 빈 리스트 반환
    if not all_predictions:
        return []
    
    # 좌표, 점수, 클래스 ID 분리
    boxes = np.array([p[:4] for p in all_predictions])
    scores = np.array([p[4] for p in all_predictions])
    class_ids = np.array([p[5] for p in all_predictions])
    
    # NMS 적용하여 중복 제거
    final_boxes, final_scores, final_class_ids = non_max_suppression(
        boxes, scores, class_ids, iou_threshold=iou_thresh,
        img_width=w, img_height=h
    )
    
    # 최종 결과 형식으로 변환
    final_predictions = []
    for i in range(len(final_boxes)):
        final_predictions.append([
            final_boxes[i][0], final_boxes[i][1], 
            final_boxes[i][2], final_boxes[i][3], 
            final_scores[i], final_class_ids[i]
        ])
    
    return final_predictions


def visualize_detection(img, boxes, class_names=None, conf_thresh=0.3, save_path=None):
    """
    객체 감지 결과를 시각화하여 표시하거나 저장
    
    매개변수:
        img: 입력 이미지 (OpenCV BGR 형식)
        boxes: 감지된 박스 목록 [x1, y1, x2, y2, confidence, class_id] 형식
        class_names: 클래스 이름 목록 (없으면 클래스 ID만 표시)
        conf_thresh: 표시할 박스의 최소 신뢰도 임계값
        save_path: 결과 이미지 저장 경로 (None이면 화면에 표시)
        
    반환값:
        시각화된 이미지 (numpy.ndarray)
    """
    # 클래스 이름이 없으면 기본값 사용
    if class_names is None:
        class_names = [
            "경차/세단", "SUV/승합차", "트럭", "버스(소형, 대형)", "통학버스(소형,대형)",
            "경찰차", "구급차", "소방차", "견인차", "기타 특장차",
            "성인", "어린이", "오토바이", "자전거 / 기타 전동 이동체",
            "라바콘", "삼각대", "기타"
        ]
    
    # 이미지 복사
    vis_img = img.copy()
    
    # 각 클래스별 색상 생성 (BGR 형식)
    colors = {}
    np.random.seed(42)  # 색상 일관성 유지
    for i in range(max(17, len(class_names))):
        colors[i] = tuple(map(int, np.random.randint(0, 255, 3)))
    
    # 박스 표시
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4]
        class_id = int(box[5]) if len(box) > 5 else 0
        
        # 임계값 미만 박스는 건너뛰기
        if conf < conf_thresh:
            continue
        
        # 박스 색상 선택
        color = colors.get(class_id, (0, 255, 0))  # 기본값: 녹색
        
        # 바운딩 박스 그리기
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # 클래스 이름과 신뢰도 표시
        label = f"{class_names[class_id]}" if class_id < len(class_names) else f"클래스 {class_id}"
        label += f": {conf:.2f}"
        
        # 라벨 배경 그리기
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        
        # 라벨 텍스트 그리기
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 이미지 저장 또는 표시
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"시각화 이미지 저장됨: {save_path}")
    
    return vis_img


def run_ensemble():
    """
    여러 모델로 앙상블 예측을 실행하고 결과 평가
    """
    print("\n=== 앙상블 예측 실행 ===")
    
    # 모델 경로 설정 (학습된 모델들)
    model_paths = [
        "runs/detect/train/weights/best.pt",
        "runs/detect/train2/weights/best.pt"  # 가능하면 다른 모델 추가
    ]
    
    # 사용 가능한 모델만 필터링
    available_models = []
    for path in model_paths:
        if os.path.exists(path):
            available_models.append(path)
    
    if len(available_models) == 0:
        print("사용 가능한 모델이 없습니다. 앙상블을 건너뜁니다.")
        return
    elif len(available_models) == 1:
        print(f"하나의 모델만 사용 가능: {available_models[0]}")
        print("단일 모델로 앙상블 효과를 얻을 수 없지만 계속 진행합니다.")
    
    print(f"사용할 모델: {available_models}")
    
    # 모델 로드
    models = []
    for path in available_models:
        try:
            model = YOLOModel()
            model.load(path)
            models.append(model)
            print(f"모델 로드 완료: {path}")
        except Exception as e:
            print(f"모델 로드 실패 ({path}): {str(e)}")
    
    # 테스트 이미지 디렉토리
    test_dir = "data/test/images"
    out_dir = "data/test/labels_ensemble"
    os.makedirs(out_dir, exist_ok=True)
    
    # 테스트 이미지 목록
    img_files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    
    if not img_files:
        print(f"테스트 이미지가 없습니다: {test_dir}")
        return
    
    print(f"테스트 이미지 수: {len(img_files)}")
    
    # 각 이미지에 대해 앙상블 예측 수행
    for i, img_path in enumerate(img_files):
        if i % 10 == 0:
            print(f"처리 중: {i+1}/{len(img_files)}")
        
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지를 로드할 수 없습니다: {img_path}")
            continue
        
        # 이미지 크기 (YOLO 좌표 정규화에 사용)
        h, w = img.shape[:2]
        
        # 앙상블 예측
        ensemble_results = ensemble_predict(models, img, conf_thresh=0.2, iou_thresh=0.45)
        
        # 결과 저장 (YOLO 형식)
        base_name = os.path.basename(img_path)
        base_name = os.path.splitext(base_name)[0]
        out_path = os.path.join(out_dir, f"{base_name}.txt")
        
        with open(out_path, 'w') as f:
            for box in ensemble_results:
                x1, y1, x2, y2, conf, class_id = box
                
                # 픽셀 좌표를 YOLO 형식으로 변환
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # YOLO 형식으로 저장
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
    
    # 결과 평가
    gt_dir = "data/test/labels"
    print("\n=== 앙상블 예측 결과 평가 ===")
    
    from .evaluate import evaluate_detection
    metrics = evaluate_detection(gt_dir, out_dir, debug=True)
    
    print(f"\n앙상블 평가 결과:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")


def run_tta():
    """
    Test-Time Augmentation을 실행하고 결과 평가
    """
    print("\n=== TTA 예측 실행 ===")
    
    # 모델 경로
    model_path = "runs/detect/train/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"모델 파일이 없습니다: {model_path}")
        return
    
    # 모델 로드
    try:
        model = YOLOModel()
        model.load(model_path)
        print(f"모델 로드 완료: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        return
    
    # 테스트 이미지 디렉토리
    test_dir = "data/test/images"
    out_dir = "data/test/labels_tta"
    os.makedirs(out_dir, exist_ok=True)
    
    # 테스트 이미지 목록
    img_files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    
    if not img_files:
        print(f"테스트 이미지가 없습니다: {test_dir}")
        return
    
    print(f"테스트 이미지 수: {len(img_files)}")
    
    # 증강 기법 목록
    augmentations = ['original', 'flip_h', 'flip_v', 'rotate90']
    
    # 각 이미지에 대해 TTA 수행
    for i, img_path in enumerate(img_files):
        if i % 10 == 0:
            print(f"처리 중: {i+1}/{len(img_files)}")
        
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지를 로드할 수 없습니다: {img_path}")
            continue
        
        # 이미지 크기
        h, w = img.shape[:2]
        
        # TTA 예측
        tta_results = test_time_augmentation(
            model, img, augmentations=augmentations, 
            conf_thresh=0.2, iou_thresh=0.45
        )
        
        # 결과 저장 (YOLO 형식)
        base_name = os.path.basename(img_path)
        base_name = os.path.splitext(base_name)[0]
        out_path = os.path.join(out_dir, f"{base_name}.txt")
        
        with open(out_path, 'w') as f:
            for box in tta_results:
                x1, y1, x2, y2, conf, class_id = box
                
                # 픽셀 좌표를 YOLO 형식으로 변환
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # YOLO 형식으로 저장
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
    
    # 결과 평가
    gt_dir = "data/test/labels"
    print("\n=== TTA 예측 결과 평가 ===")
    
    from .evaluate import evaluate_detection
    metrics = evaluate_detection(gt_dir, out_dir, debug=True)
    
    print(f"\nTTA 평가 결과:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")