import numpy as np
import cv2
import os
from .model_yolo import YOLOModel

def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.25):
    """중복 박스를 제거하는 NMS 알고리즘"""
    if len(boxes) == 0:
        return []
        
    # 신뢰도 기준 필터링
    filtered_boxes = [b for b in boxes if b[4] >= conf_threshold]
    if len(filtered_boxes) == 0:
        return []
    
    # 클래스별로 처리
    class_boxes = {}
    for box in filtered_boxes:
        class_id = int(box[5])
        if class_id not in class_boxes:
            class_boxes[class_id] = []
        class_boxes[class_id].append(box)
    
    # 클래스별 NMS 적용
    result_boxes = []
    for class_id, boxes in class_boxes.items():
        # 신뢰도 기준 내림차순 정렬
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        picked = []
        
        while len(boxes) > 0:
            # 신뢰도 최대 박스 선택
            current = boxes[0]
            picked.append(current)
            
            if len(boxes) == 1:
                break
                
            # 남은 박스들
            rest = boxes[1:]
            boxes = []
            
            for box in rest:
                # IoU 계산
                x1 = max(current[0], box[0])
                y1 = max(current[1], box[1])
                x2 = min(current[2], box[2])
                y2 = min(current[3], box[3])
                
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                
                intersection = w * h
                box1_area = (current[2] - current[0]) * (current[3] - current[1])
                box2_area = (box[2] - box[0]) * (box[3] - box[1])
                union = box1_area + box2_area - intersection
                
                iou = intersection / union if union > 0 else 0
                
                # IoU가 임계값보다 작으면 유지
                if iou < iou_threshold:
                    boxes.append(box)
        
        result_boxes.extend(picked)
    
    return result_boxes

def ensemble_predict(models, img, conf_thresh=0.2, iou_thresh=0.5):
    """여러 모델의 예측을 결합"""
    all_boxes = []
    for i, model in enumerate(models):
        print(f"모델 {i+1}/{len(models)} 예측 중...")
        boxes = model.predict(img, conf_thresh=conf_thresh)
        all_boxes.extend(boxes)
    
    # NMS로 중복 박스 제거
    final_boxes = non_max_suppression(all_boxes, iou_threshold=iou_thresh)
    print(f"앙상블 결과: {len(final_boxes)}개 객체 감지됨")
    return final_boxes

def transform_boxes(boxes, orig_shape, aug_shape):
    """증강된 이미지의 박스 좌표를 원본 이미지 좌표로 변환"""
    if not boxes:
        return []
        
    # 스케일 계산
    orig_h, orig_w = orig_shape[:2]
    aug_h, aug_w = aug_shape[:2]
    w_scale = orig_w / aug_w
    h_scale = orig_h / aug_h
    
    transformed = []
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        # 좌표 변환
        x1 = x1 * w_scale
        y1 = y1 * h_scale
        x2 = x2 * w_scale
        y2 = y2 * h_scale
        
        # 범위 제한
        x1 = max(0, min(orig_w, x1))
        y1 = max(0, min(orig_h, y1))
        x2 = max(0, min(orig_w, x2))
        y2 = max(0, min(orig_h, y2))
        
        transformed.append([x1, y1, x2, y2, conf, class_id])
    
    return transformed

def test_time_augmentation(model, img, augment_count=5):
    """테스트 시 이미지 증강하여 예측 결과 개선"""
    print("TTA 시작: 원본 이미지 예측...")
    # 원본 이미지 예측
    boxes = model.predict(img)
    
    # 다양한 증강 적용
    augmentations = [
        lambda x: cv2.flip(x, 1),  # 좌우 반전
        lambda x: cv2.resize(x, (int(x.shape[1]*0.8), int(x.shape[0]*0.8))),  # 축소
        lambda x: cv2.resize(x, (int(x.shape[1]*1.2), int(x.shape[0]*1.2))),  # 확대
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),  # 90도 회전
        lambda x: cv2.addWeighted(x, 1.5, x, 0, 0)  # 대비 증가
    ]
    
    all_boxes = boxes.copy()
    print(f"원본 이미지에서 {len(boxes)}개 객체 감지됨")
    
    # 실제 사용할 증강 수 제한
    used_augmentations = augmentations[:min(augment_count, len(augmentations))]
    
    for i, aug_func in enumerate(used_augmentations):
        print(f"증강 {i+1}/{len(used_augmentations)} 적용 중...")
        # 증강 적용
        aug_img = aug_func(img.copy())
        # 예측
        aug_boxes = model.predict(aug_img)
        # 원본 이미지 좌표로 변환
        transformed_boxes = transform_boxes(aug_boxes, img.shape, aug_img.shape)
        all_boxes.extend(transformed_boxes)
        print(f"  - 증강 {i+1}에서 {len(aug_boxes)}개 객체 감지됨")
    
    # NMS로 중복 박스 제거
    final_boxes = non_max_suppression(all_boxes, iou_threshold=0.5)
    print(f"TTA 최종 결과: {len(final_boxes)}개 객체 감지됨")
    return final_boxes

def visualize_detection(img, boxes, class_names=None, save_path=None):
    """탐지 결과를 시각화"""
    img_copy = img.copy()
    
    # 색상 설정 (클래스별로 다른 색상)
    colors = [
        (0, 255, 0),    # 녹색
        (255, 0, 0),    # 파란색
        (0, 0, 255),    # 빨간색
        (255, 255, 0),  # 청록색
        (255, 0, 255),  # 자주색
        (0, 255, 255),  # 노란색
        (128, 128, 0),  # 올리브색
        (128, 0, 128),  # 보라색
        (0, 128, 128),  # 테일색
    ]
    
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        class_id = int(class_id)
        
        # 좌표를 정수로 변환
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # 색상 선택
        color = colors[class_id % len(colors)]
        
        # 박스 그리기
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # 클래스 이름과 신뢰도
        label = f"{class_names[class_id] if class_names else f'Class {class_id}'}: {conf:.2f}"
        
        # 텍스트 크기 계산
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 텍스트 배경 그리기
        cv2.rectangle(img_copy, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # 텍스트 그리기
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 이미지 저장
    if save_path:
        cv2.imwrite(save_path, img_copy)
        print(f"시각화된 이미지 저장됨: {save_path}")
    
    return img_copy

def run_ensemble():
    print("\n=== 앙상블 모델 실행 ===")
    # 여러 크기의 모델 로드
    models = []
    
    # 여러 모델 로드 (예: nano와 small 모델)
    model_sizes = ['n', 's']  # 사용 가능한 모델 크기
    for size in model_sizes:
        model_path = f"runs/detect/train_{size}/weights/best.pt"
        if os.path.exists(model_path):
            print(f"모델 로드 중: {model_path}")
            model = YOLOModel()
            model.load(model_path)
            models.append(model)
        else:
            print(f"모델 파일을 찾을 수 없음: {model_path}")
    
    if not models:
        print("사용 가능한 모델이 없습니다. 확인 후 다시 시도하세요.")
        return
    
    # 테스트 이미지 선택 (첫 번째 테스트 이미지 사용)
    test_img_files = os.listdir("data/test/images")
    test_img_files = [f for f in test_img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_img_files:
        print("테스트 이미지를 찾을 수 없습니다.")
        return
        
    test_img_path = os.path.join("data/test/images", test_img_files[0])
    print(f"테스트 이미지: {test_img_path}")
    
    img = cv2.imread(test_img_path)
    ensemble_boxes = ensemble_predict(models, img)
    
    # 결과 시각화
    class_names = [
        "경차/세단", "SUV/승합차", "트럭", "버스(소형, 대형)",
        "통학버스(소형,대형)", "경찰차", "구급차", "소방차",
        "견인차", "기타 특장차", "성인", "어린이",
        "오토바이", "자전거 / 기타 전동 이동체", "라바콘", "삼각대", "기타"
    ]
    
    vis_img = visualize_detection(
        img, ensemble_boxes, class_names, 
        save_path="ensemble_result.jpg"
    )
    print(f"앙상블 결과: {len(ensemble_boxes)}개 객체 감지됨, 결과 이미지: ensemble_result.jpg")
    return ensemble_boxes

def run_tta(model=None):
    """테스트 시간 증강(TTA) 실행 예시"""
    print("\n=== 테스트 시간 증강(TTA) 실행 ===")
    
    # 모델이 전달되지 않았으면 로드
    if model is None:
        model_path = "runs/detect/train/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"모델 파일을 찾을 수 없음: {model_path}")
            return
            
        model = YOLOModel()
        model.load(model_path)
    
    # 테스트 이미지 선택 (첫 번째 테스트 이미지 사용)
    test_img_files = os.listdir("data/test/images")
    test_img_files = [f for f in test_img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_img_files:
        print("테스트 이미지를 찾을 수 없습니다.")
        return
        
    test_img_path = os.path.join("data/test/images", test_img_files[0])
    print(f"테스트 이미지: {test_img_path}")
    
    img = cv2.imread(test_img_path)
    tta_boxes = test_time_augmentation(model, img)
    
    # 결과 시각화
    class_names = [
        "경차/세단", "SUV/승합차", "트럭", "버스(소형, 대형)",
        "통학버스(소형,대형)", "경찰차", "구급차", "소방차",
        "견인차", "기타 특장차", "성인", "어린이",
        "오토바이", "자전거 / 기타 전동 이동체", "라바콘", "삼각대", "기타"
    ]
    
    vis_img = visualize_detection(
        img, tta_boxes, class_names, 
        save_path="tta_result.jpg"
    )
    print(f"TTA 결과: {len(tta_boxes)}개 객체 감지됨, 결과 이미지: tta_result.jpg")
    return tta_boxes

def run_ensemble_on_directory(input_dir, output_dir):
    """디렉토리 내 모든 이미지에 앙상블 적용"""
    if not os.path.exists(input_dir):
        print(f"입력 디렉토리가 존재하지 않음: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로드
    models = []
    model_sizes = ['n', 's']
    for size in model_sizes:
        model_path = f"runs/detect/train_{size}/weights/best.pt"
        if os.path.exists(model_path):
            model = YOLOModel()
            model.load(model_path)
            models.append(model)
    
    if not models:
        print("사용 가능한 모델이 없습니다.")
        return
    
    # 클래스 이름
    class_names = [
        "경차/세단", "SUV/승합차", "트럭", "버스(소형, 대형)",
        "통학버스(소형,대형)", "경찰차", "구급차", "소방차",
        "견인차", "기타 특장차", "성인", "어린이",
        "오토바이", "자전거 / 기타 전동 이동체", "라바콘", "삼각대", "기타"
    ]
    
    # 모든 이미지 처리
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"총 {len(img_files)}개 이미지 처리 중...")
    
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(input_dir, img_file)
        out_path = os.path.join(output_dir, f"ensemble_{img_file}")
        
        print(f"[{i+1}/{len(img_files)}] {img_file} 처리 중...")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"이미지를 읽을 수 없음: {img_path}")
            continue
        
        # 앙상블 예측 수행
        boxes = ensemble_predict(models, img)
        
        # 결과 시각화 및 저장
        visualize_detection(img, boxes, class_names, save_path=out_path)
    
    print(f"앙상블 처리 완료. 결과 저장 경로: {output_dir}")