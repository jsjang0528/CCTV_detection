import glob
import os

import numpy as np


def calculate_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다.

    매개변수:
        box1 (list/ndarray): [x1, y1, x2, y2] 형식의 첫 번째 박스
        box2 (list/ndarray): [x1, y1, x2, y2] 형식의 두 번째 박스

    반환값:
        float: IoU 값 (0~1)
    """
    # box2가 다중 박스인 경우 calculate_ious 함수를 호출
    if isinstance(box2, np.ndarray) and box2.ndim > 1:
        return calculate_ious(box1, box2)

    # 단일 박스 간 IoU 계산
    box1_x1, box1_y1, box1_x2, box1_y2 = box1

    # box2가 리스트 또는 1차원 배열인 경우
    if isinstance(box2, (list, tuple)) or (
        isinstance(box2, np.ndarray) and box2.ndim == 1
    ):
        box2_x1, box2_y1, box2_x2, box2_y2 = box2
    else:
        raise ValueError(f"지원되지 않는 box2 형식: {type(box2)}, 값: {box2}")

    # 교차 영역 계산
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)

    # 교차 영역이 없는 경우
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 교차 영역 넓이
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 두 박스 넓이
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # IoU = 교차 영역 / 합집합 영역
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return max(0.0, min(iou, 1.0))  # 0~1 범위로 제한


def calculate_ious(box, boxes):
    """
    하나의 박스와 여러 박스 간의 IoU를 한 번에 계산합니다.

    매개변수:
        box (list/ndarray): [x1, y1, x2, y2] 형식의 기준 박스
        boxes (ndarray): N개의 박스로 이루어진 배열 [N, 4]

    반환값:
        ndarray: N개의 IoU 값 배열
    """
    # 입력 확인
    if not isinstance(boxes, np.ndarray) or boxes.ndim != 2:
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        elif isinstance(boxes, np.ndarray) and boxes.ndim == 1:
            # 단일 박스인 경우 2차원으로 변환
            boxes = np.array([boxes])
        else:
            return np.array([calculate_iou(box, b) for b in boxes])

    # boxes가 비어있는 경우
    if boxes.shape[0] == 0:
        return np.array([])

    # 기준 박스
    box_x1, box_y1, box_x2, box_y2 = box
    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)

    # 모든 박스
    boxes_x1 = boxes[:, 0]
    boxes_y1 = boxes[:, 1]
    boxes_x2 = boxes[:, 2]
    boxes_y2 = boxes[:, 3]
    boxes_area = (boxes_x2 - boxes_x1) * (boxes_y2 - boxes_y1)

    # 교차 영역 계산
    x_left = np.maximum(box_x1, boxes_x1)
    y_top = np.maximum(box_y1, boxes_y1)
    x_right = np.minimum(box_x2, boxes_x2)
    y_bottom = np.minimum(box_y2, boxes_y2)

    # 교차 영역 넓이 (음수 방지)
    intersection_width = np.maximum(0, x_right - x_left)
    intersection_height = np.maximum(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    # IoU = 교차 영역 / 합집합 영역
    union_area = box_area + boxes_area - intersection_area
    ious = np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(union_area, dtype=float),
        where=union_area != 0,
    )

    return ious


def evaluate_detection(gt_dir, pred_dir, debug=False):
    """
    객체 감지 모델의 성능을 평가합니다.

    매개변수:
        gt_dir (str): 정답 라벨이 있는 디렉토리 경로
        pred_dir (str): 예측 라벨이 있는 디렉토리 경로
        debug (bool): 디버깅 정보 출력 여부

    반환값:
        dict: 평가 지표 (AP, Precision, Recall)
    """
    # 디버깅 메시지
    if debug:
        print(f"\n=== 평가 시작 ===")
        print(f"정답 디렉토리: {gt_dir}")
        print(f"예측 디렉토리: {pred_dir}")

    # 정답 및 예측 라벨 파일 목록
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))

    if not gt_files:
        print(f"경고: 정답 라벨 파일이 없습니다 - {gt_dir}")
        return {"AP": 0.0, "Precision": 0.0, "Recall": 0.0}

    if debug:
        print(f"정답 파일 수: {len(gt_files)}")
        print(f"첫 번째 정답 파일: {gt_files[0]}")

    # 이미지 크기 정보를 가져올 디렉토리
    img_dir = os.path.join(os.path.dirname(gt_dir), "images")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(os.path.dirname(gt_dir), "images_processed")
        if not os.path.exists(img_dir):
            print(
                f"경고: 이미지 디렉토리를 찾을 수 없습니다. 기본 이미지 크기(640x480)를 사용합니다."
            )
            img_dir = None

    if debug and img_dir:
        print(f"이미지 디렉토리: {img_dir}")
        img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(
            os.path.join(img_dir, "*.png")
        )
        print(f"이미지 파일 수: {len(img_files)}")

    all_gt_boxes = []
    all_pred_boxes = []

    # 기본 이미지 크기
    default_img_width = 640
    default_img_height = 480

    # 총 박스 수 카운트
    total_gt_boxes = 0
    total_pred_boxes = 0

    # 각 라벨 파일에 대해
    for i, gt_file in enumerate(gt_files):
        basename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, basename)

        # 디버깅 (10개 파일마다 1개씩 출력)
        debug_this_file = debug and (i < 5 or i % 100 == 0)

        if debug_this_file:
            print(f"\n처리 중: {basename} ({i+1}/{len(gt_files)})")

        # 예측 파일이 없으면 빈 예측으로 처리
        if not os.path.exists(pred_file):
            if debug_this_file:
                print(f"경고: 예측 파일이 없습니다 - {basename}")
            all_pred_boxes.append([])
            continue

        # 실제 이미지 크기 가져오기
        img_width, img_height = default_img_width, default_img_height
        if img_dir:
            img_name = os.path.splitext(basename)[0]
            img_path = None

            # 확장자 찾기
            for ext in [".jpg", ".png", ".jpeg"]:
                temp_path = os.path.join(img_dir, img_name + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break

            if img_path and os.path.exists(img_path):
                try:
                    import cv2

                    img = cv2.imread(img_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        if debug_this_file:
                            print(f"이미지 크기: {img_width}x{img_height}")
                    else:
                        if debug_this_file:
                            print(f"경고: 이미지를 로드할 수 없습니다 - {img_path}")
                except Exception as e:
                    if debug_this_file:
                        print(f"이미지 로드 실패 ({img_path}): {e}")
            elif debug_this_file:
                print(
                    f"경고: 이미지를 찾을 수 없습니다. 기본 크기 사용: {img_width}x{img_height}"
                )

        # 정답 바운딩 박스 로드
        gt_boxes = []
        try:
            with open(gt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])

                        # YOLO 형식 → 픽셀 좌표 변환
                        x_min = max(0, int((x_c - w / 2) * img_width))
                        y_min = max(0, int((y_c - h / 2) * img_height))
                        x_max = min(img_width, int((x_c + w / 2) * img_width))
                        y_max = min(img_height, int((y_c + h / 2) * img_height))

                        # 유효한 박스인지 확인 (너비와 높이가 0보다 커야 함)
                        if x_max > x_min and y_max > y_min:
                            gt_boxes.append([x_min, y_min, x_max, y_max, class_id])
                        elif debug_this_file:
                            print(
                                f"경고: 유효하지 않은 GT 박스 무시 - {[x_min, y_min, x_max, y_max]}"
                            )
        except Exception as e:
            if debug:
                print(f"GT 파일 로드 실패 ({gt_file}): {e}")

        # 예측 바운딩 박스 로드
        pred_boxes = []
        try:
            with open(pred_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if (
                        len(parts) >= 6
                    ):  # 클래스, x_center, y_center, width, height, confidence
                        class_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        confidence = float(parts[5])

                        # YOLO 형식 → 픽셀 좌표 변환
                        x_min = max(0, int((x_c - w / 2) * img_width))
                        y_min = max(0, int((y_c - h / 2) * img_height))
                        x_max = min(img_width, int((x_c + w / 2) * img_width))
                        y_max = min(img_height, int((y_c + h / 2) * img_height))

                        # 유효한 박스인지 확인 (너비와 높이가 0보다 커야 함)
                        if (
                            x_max > x_min and y_max > y_min and confidence > 0.01
                        ):  # 낮은 임계값
                            pred_boxes.append(
                                [x_min, y_min, x_max, y_max, confidence, class_id]
                            )
                        elif debug_this_file:
                            print(
                                f"경고: 유효하지 않은 예측 박스 무시 - {[x_min, y_min, x_max, y_max]}"
                            )
        except Exception as e:
            if debug:
                print(f"예측 파일 로드 실패 ({pred_file}): {e}")

        # 박스 수 카운트
        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)

        # 디버깅 정보
        if debug_this_file:
            print(f"정답 박스 수: {len(gt_boxes)}")
            print(f"예측 박스 수: {len(pred_boxes)}")
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                # 첫 번째 박스만 출력
                print(f"첫 번째 정답 박스: {gt_boxes[0]}")
                print(f"첫 번째 예측 박스: {pred_boxes[0]}")

        all_gt_boxes.append(gt_boxes)
        all_pred_boxes.append(pred_boxes)

    # 박스 수 집계
    if debug:
        print(f"\n총 파일 수: {len(gt_files)}")
        print(f"총 정답 박스 수: {total_gt_boxes}")
        print(f"총 예측 박스 수: {total_pred_boxes}")

        if total_gt_boxes == 0:
            print("경고: 정답 박스가 없습니다!")
        if total_pred_boxes == 0:
            print("경고: 예측 박스가 없습니다!")

    # 박스가 하나도 없으면 평가 불가능
    if total_gt_boxes == 0 or total_pred_boxes == 0:
        print("박스가 없어 정확한 평가가 불가능합니다.")
        return {"AP": 0.0, "Precision": 0.0, "Recall": 0.0}

    # 클래스 이름 목록
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

    # IoU 임계값 설정
    iou_threshold = 0.5  # 기본 IoU 임계값

    # 각 클래스별 TP, FP, FN 초기화
    # True Positive (올바르게 감지된 객체)
    # False Positive (잘못 감지된 객체)
    # False Negative (감지되지 않은 객체)
    class_metrics = {i: {"TP": 0, "FP": 0, "FN": 0} for i in range(len(class_names))}
    total_metrics = {"TP": 0, "FP": 0, "FN": 0}

    # 각 이미지에 대한 평가 수행
    for idx, (gt_boxes, pred_boxes) in enumerate(zip(all_gt_boxes, all_pred_boxes)):
        # 디버깅 정보
        debug_this_image = debug and (idx < 5 or idx % 100 == 0)
        if debug_this_image:
            print(f"\n이미지 {idx+1}/{len(all_gt_boxes)} 평가 중...")
            print(f"정답 박스: {len(gt_boxes)}, 예측 박스: {len(pred_boxes)}")

        # 정답이 없는 경우 - 모든 예측은 FP
        if len(gt_boxes) == 0:
            for pred_box in pred_boxes:
                class_id = int(pred_box[5]) if len(pred_box) > 5 else 0
                class_metrics[class_id]["FP"] += 1
                total_metrics["FP"] += 1
            continue

        # 예측이 없는 경우 - 모든 정답은 FN
        if len(pred_boxes) == 0:
            for gt_box in gt_boxes:
                class_id = int(gt_box[4]) if len(gt_box) > 4 else 0
                class_metrics[class_id]["FN"] += 1
                total_metrics["FN"] += 1
            continue

        # 정답 박스 매칭 상태 추적
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        # 예측 박스를 신뢰도 점수로 정렬 (내림차순)
        pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        # 각 예측 박스에 대해 처리
        for pred_box in pred_boxes_sorted:
            x1_p, y1_p, x2_p, y2_p = pred_box[:4]
            confidence = pred_box[4]
            class_id = int(pred_box[5]) if len(pred_box) > 5 else 0

            # 최대 IoU와 해당 정답 박스 찾기
            max_iou = -1
            max_iou_idx = -1

            for i, gt_box in enumerate(gt_boxes):
                # 이미 매칭된 정답 박스는 건너뛰기
                if gt_matched[i]:
                    continue

                # 정답 박스 클래스 확인
                gt_class = int(gt_box[4]) if len(gt_box) > 4 else 0

                # 클래스가 다르면 건너뛰기
                if gt_class != class_id:
                    continue

                # IoU 계산
                x1_g, y1_g, x2_g, y2_g = gt_box[:4]
                iou = calculate_iou([x1_p, y1_p, x2_p, y2_p], [x1_g, y1_g, x2_g, y2_g])

                # 최대 IoU 갱신
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = i

            # 충분한 IoU를 가진 매칭이 있는 경우 TP, 아니면 FP
            if max_iou >= iou_threshold and max_iou_idx >= 0:
                gt_matched[max_iou_idx] = True  # 정답 박스 매칭 표시
                class_metrics[class_id]["TP"] += 1
                total_metrics["TP"] += 1
                if debug_this_image:
                    print(
                        f"TP: 클래스 {class_id}, 신뢰도 {confidence:.3f}, IoU {max_iou:.3f}"
                    )
            else:
                class_metrics[class_id]["FP"] += 1
                total_metrics["FP"] += 1
                if debug_this_image:
                    print(
                        f"FP: 클래스 {class_id}, 신뢰도 {confidence:.3f}, 최대 IoU {max_iou:.3f}"
                    )

        # 매칭되지 않은 정답 박스는 FN으로 처리
        for i, matched in enumerate(gt_matched):
            if not matched:
                gt_class = int(gt_boxes[i][4]) if len(gt_boxes[i]) > 4 else 0
                class_metrics[gt_class]["FN"] += 1
                total_metrics["FN"] += 1
                if debug_this_image:
                    print(f"FN: 클래스 {gt_class}, 박스 {gt_boxes[i][:4]}")

    # 정밀도와 재현율 계산
    precision = (
        total_metrics["TP"] / (total_metrics["TP"] + total_metrics["FP"])
        if (total_metrics["TP"] + total_metrics["FP"]) > 0
        else 0
    )
    recall = (
        total_metrics["TP"] / (total_metrics["TP"] + total_metrics["FN"])
        if (total_metrics["TP"] + total_metrics["FN"]) > 0
        else 0
    )

    # F1 점수 계산
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    if debug:
        print(f"\n=== 최종 결과 ===")
        print(f"True Positives: {total_metrics['TP']}")
        print(f"False Positives: {total_metrics['FP']}")
        print(f"False Negatives: {total_metrics['FN']}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # 클래스별 결과 출력
        print("\n=== 클래스별 결과 ===")
        for class_id, metrics in class_metrics.items():
            if (
                metrics["TP"] + metrics["FP"] + metrics["FN"] > 0
            ):  # 해당 클래스가 존재하는 경우만
                class_precision = (
                    metrics["TP"] / (metrics["TP"] + metrics["FP"])
                    if (metrics["TP"] + metrics["FP"]) > 0
                    else 0
                )
                class_recall = (
                    metrics["TP"] / (metrics["TP"] + metrics["FN"])
                    if (metrics["TP"] + metrics["FN"]) > 0
                    else 0
                )
                class_f1 = (
                    2
                    * class_precision
                    * class_recall
                    / (class_precision + class_recall)
                    if (class_precision + class_recall) > 0
                    else 0
                )
                print(
                    f"클래스 {class_id} ({class_names[class_id]}): TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}"
                )
                print(
                    f"  Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, F1: {class_f1:.4f}"
                )

    # 기본 메트릭 초기화
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": total_metrics["TP"],
        "FP": total_metrics["FP"],
        "FN": total_metrics["FN"],
    }

    # mAP 계산 (단일 IoU 임계값 및 다중 임계값)
    if debug:
        print("\n=== mAP 계산 중... ===")

    # 다양한 IoU 임계값에서 mAP 계산
    iou_thresholds = [
        0.5
    ]  # 필요시 [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] 추가
    map_results = calculate_map(
        all_gt_boxes, all_pred_boxes, class_names, iou_thresholds, debug
    )

    # 메트릭에 mAP 결과 추가
    metrics.update(map_results)

    # 디버깅 정보 출력
    if debug:
        print("\n=== 최종 평가 결과 ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"mAP@0.5: {map_results.get('mAP@0.5', 0):.4f}")

    return metrics


def calculate_map(
    all_gt_boxes, all_pred_boxes, class_names, iou_thresholds=[0.5], debug=False
):
    """
    AP(Average Precision)와 mAP(mean Average Precision)를 계산합니다.

    매개변수:
        all_gt_boxes (list): 모든 이미지의 정답 바운딩 박스 목록
        all_pred_boxes (list): 모든 이미지의 예측 바운딩 박스 목록
        class_names (list): 클래스 이름 목록
        iou_thresholds (list): 사용할 IoU 임계값 목록 (기본값: [0.5])
        debug (bool): 디버깅 정보 출력 여부

    반환값:
        dict: mAP 및 각 클래스별 AP 값을 포함하는 딕셔너리
    """
    # 결과 저장 딕셔너리
    results = {}

    # 각 IoU 임계값에 대한 mAP 계산
    for iou_threshold in iou_thresholds:
        # 클래스별 예측 결과 저장
        class_predictions = {i: [] for i in range(len(class_names))}
        class_gt_count = {i: 0 for i in range(len(class_names))}

        # 모든 이미지에 대해 처리
        for img_idx, (gt_boxes, pred_boxes) in enumerate(
            zip(all_gt_boxes, all_pred_boxes)
        ):
            # 정답 박스 클래스별 카운트
            for gt_box in gt_boxes:
                if len(gt_box) > 4:  # 클래스 정보가 있는 경우
                    gt_class = int(gt_box[4])
                    class_gt_count[gt_class] += 1
                else:  # 클래스 정보가 없는 경우 (기본: 클래스 0)
                    class_gt_count[0] += 1

            # 각 클래스별 정답 박스 추출 및 매칭 상태 초기화
            class_gt_boxes = {}
            class_gt_matched = {}

            for class_id in range(len(class_names)):
                # 현재 클래스의 정답 박스 추출
                current_gt_boxes = []
                for gt_box in gt_boxes:
                    if len(gt_box) > 4 and int(gt_box[4]) == class_id:
                        current_gt_boxes.append(gt_box)
                    elif (
                        len(gt_box) <= 4 and class_id == 0
                    ):  # 클래스 정보가 없는 경우 클래스 0으로 간주
                        current_gt_boxes.append(gt_box)

                class_gt_boxes[class_id] = current_gt_boxes
                class_gt_matched[class_id] = [False] * len(current_gt_boxes)

            # 예측 결과 처리 (모든 클래스에 대해)
            for pred_box in pred_boxes:
                if len(pred_box) < 6:  # 신뢰도와 클래스 정보가 필요
                    continue

                confidence = pred_box[4]
                class_id = int(pred_box[5])

                # 현재 클래스의 정답 박스들
                current_gt_boxes = class_gt_boxes.get(class_id, [])
                current_gt_matched = class_gt_matched.get(class_id, [])

                if not current_gt_boxes:  # 해당 클래스의 정답 박스가 없음 (FP)
                    class_predictions[class_id].append([confidence, 0])
                    continue

                # 최대 IoU 및 인덱스 찾기
                max_iou = -1
                max_idx = -1

                for gt_idx, gt_box in enumerate(current_gt_boxes):
                    if current_gt_matched[gt_idx]:  # 이미 매칭된 정답은 건너뛰기
                        continue

                    # IoU 계산
                    pred_coords = pred_box[:4]
                    gt_coords = gt_box[:4]

                    iou = calculate_iou(pred_coords, gt_coords)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = gt_idx

                # 예측 정보 저장 (신뢰도, 정답 여부)
                is_correct = 0
                if max_iou >= iou_threshold and max_idx >= 0:
                    is_correct = 1
                    current_gt_matched[max_idx] = True  # 정답 박스 매칭 표시

                # 클래스별로 예측 결과 저장 [신뢰도, 정답여부(1/0)]
                class_predictions[class_id].append([confidence, is_correct])

        # 각 클래스별 AP 계산
        aps = {}

        for class_id in range(len(class_names)):
            predictions = class_predictions[class_id]

            # 예측이 없거나 정답이 없는 경우 AP = 0
            if len(predictions) == 0 or class_gt_count.get(class_id, 0) == 0:
                aps[class_id] = 0.0
                continue

            # 신뢰도 내림차순 정렬
            predictions.sort(key=lambda x: x[0], reverse=True)

            # precision, recall 배열 계산
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []

            for i, (_, is_correct) in enumerate(predictions):
                if is_correct:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1

                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / class_gt_count[class_id]

                precisions.append(precision)
                recalls.append(recall)

            # 11-point 방식의 AP 계산 (PASCAL VOC 방식)
            ap = 0
            for t in np.arange(0, 1.1, 0.1):  # 0, 0.1, 0.2, ..., 1.0
                p_vals = [p for r, p in zip(recalls, precisions) if r >= t]
                p_at_t = max(p_vals) if p_vals else 0
                ap += p_at_t / 11

            aps[class_id] = ap

        # 클래스별 AP 계산 결과에서 NaN이나 None 값 처리
        valid_aps = [ap for ap in aps.values() if ap is not None and not np.isnan(ap)]

        # mAP 계산 (평균 AP)
        mean_ap = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0

        # 결과 저장
        results[f"mAP@{iou_threshold}"] = mean_ap
        results[f"AP_by_class@{iou_threshold}"] = aps

        # 디버깅 정보 출력
        if debug:
            print(f"\n=== mAP@{iou_threshold} 계산 결과 ===")
            print(f"mAP: {mean_ap:.4f}")

            print("\n클래스별 AP:")
            for class_id, ap in aps.items():
                if (
                    class_gt_count.get(class_id, 0) > 0
                ):  # 해당 클래스가 데이터셋에 존재하는 경우만
                    print(
                        f"클래스 {class_id} ({class_names[class_id]}): AP = {ap:.4f} (GT 수: {class_gt_count.get(class_id, 0)}, 예측 수: {len(class_predictions[class_id])})"
                    )

    return results
