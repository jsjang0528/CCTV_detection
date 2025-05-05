import glob
import json
import os

import cv2


def basic_image_preprocess(
    input_dir,
    output_dir,
    use_gray=True,
    use_clahe=True,
    overwrite=False,
):
    """
    CCTV 이미지에 대한 기본적인 전처리를 수행합니다.

    매개변수:
        input_dir (str): 원본 이미지가 저장된 디렉토리 경로
        output_dir (str): 전처리된 이미지를 저장할 디렉토리 경
        use_gray (bool): 그레이스케일 변환 여부
        use_clahe (bool): CLAHE(대비 제한 적응형 히스토그램 평활화) 적용 여부
        overwrite (bool): 이미 존재하는 파일을 덮어쓸지 여부
            - True: 모든 이미지를 다시 처리
            - False: 이미 처리된 이미지는 건너뜀 (기본값)

    동작:
        1. 입력 디렉토리에서 모든 JPG 이미지를 로드
        2. 필요시 그레이스케일 변환 및 CLAHE 적용
        3. 지정된 크기로 이미지 리사이즈
        4. 결과 이미지를 출력 디렉토리에 저장
    """
    os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리 생성 (없으면)
    img_files = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg"))
    )  # 모든 JPG 파일 가져오기

    processed_count = 0
    skipped_count = 0

    for f in img_files:
        # 이미지 로드
        img = cv2.imread(f)
        if img is None:
            print(f"[경고] 이미지 읽기 실패: {f}")
            continue

        # 그레이스케일 변환 (선택 사항)
        if use_gray:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # CLAHE 적용 (선택 사항)
        if use_clahe:
            # YCrCb 색상 공간으로 변환 (Y: 밝기 채널)
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            # 밝기 채널에만 CLAHE 적용
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y = clahe.apply(y)
            # 채널 합치기
            merged = cv2.merge((y, cr, cb))
            img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        # 결과 이미지 저장
        base = os.path.basename(f)
        out_path = os.path.join(output_dir, base)

        # overwrite=False이고 이미 파일이 존재하면 건너뜀
        if not overwrite and os.path.exists(out_path):
            skipped_count += 1
            continue

        cv2.imwrite(out_path, img)
        processed_count += 1

    print(
        f"[basic_image_preprocess] 전처리 완료 => {output_dir}, 처리된 이미지 수: {processed_count}, 생략된 이미지 수: {skipped_count}"
    )
    return processed_count


def create_yolo_labels(img_dir, json_dir, out_dir, overwrite=False):
    """
    JSON 형식의 어노테이션을 YOLO 형식 라벨 파일로 변환합니다.
    모든 객체 유형을 감지합니다.
    """
    import json
    import os

    import cv2

    # 클래스 매핑 사전 정의
    class_mapping = {
        "경차/세단": 0,
        "SUV/승합차": 1,
        "트럭": 2,
        "버스(소형, 대형)": 3,
        "통학버스(소형,대형)": 4,
        "경찰차": 5,
        "구급차": 6,
        "소방차": 7,
        "견인차": 8,
        "기타 특장차": 9,
        "성인": 10,
        "어린이": 11,
        "오토바이": 12,
        "자전거 / 기타 전동 이동체": 13,
        "라바콘": 14,
        "삼각대": 15,
        "기타": 16,
    }

    # 디렉토리 존재 확인 및 생성
    if not os.path.exists(img_dir):
        print(f"[create_yolo_labels] 오류: 이미지 디렉토리가 존재하지 않음: {img_dir}")
        return 0

    if not os.path.exists(json_dir):
        print(f"[create_yolo_labels] 오류: JSON 디렉토리가 존재하지 않음: {json_dir}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    print(
        f"[create_yolo_labels] 디렉토리 확인 완료 - 이미지: {img_dir}, JSON: {json_dir}, 출력: {out_dir}"
    )

    # 이미지 및 JSON 파일 목록 가져오기
    img_files = {}
    for f in os.listdir(img_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            # 파일명에서 확장자 제거
            base_name = os.path.splitext(f)[0]
            img_files[base_name] = f

    json_files = {}
    for f in os.listdir(json_dir):
        if f.lower().endswith(".json"):
            # JSON 파일명에서 가능한 '.jpg' 부분과 '.json' 확장자 모두 제거
            base_name = f.replace(".jpg.json", "").replace(".json", "")
            json_files[base_name] = f

    print(
        f"[create_yolo_labels] 발견된 파일 - 이미지: {len(img_files)}개, JSON: {len(json_files)}개"
    )

    # 공통 기본 이름 찾기
    common_names = set(img_files.keys()) & set(json_files.keys())

    if not common_names:
        print("[create_yolo_labels] 오류: 매칭되는 이미지-JSON 파일 쌍이 없음")
        print(f"이미지 파일 예시: {list(img_files.keys())[:3]}")
        print(f"JSON 파일 예시: {list(json_files.keys())[:3]}")
        return 0

    print(f"[create_yolo_labels] 매칭된 파일 쌍: {len(common_names)}개")

    processed_count = 0
    skipped_count = 0
    error_count = 0
    class_counts = {i: 0 for i in range(len(class_mapping))}

    # 좌표 정규화 및 클리핑 함수 추가
    def normalize_and_clip(value, max_val):
        """좌표를 정규화하고 [0,1] 범위로 클리핑"""
        normalized = value / max_val
        return max(0.0, min(1.0, normalized))  # 0~1 범위로 클리핑

    for base_name in sorted(common_names):
        # 출력 YOLO 라벨 파일 경로
        out_file = os.path.join(out_dir, f"{base_name}.txt")

        # overwrite=False이고 파일이 이미 존재하면 건너뜀
        if not overwrite and os.path.exists(out_file):
            skipped_count += 1
            continue

        try:
            # 이미지 파일 로드
            img_path = os.path.join(img_dir, img_files[base_name])
            img = cv2.imread(img_path)
            if img is None:
                print(f"[create_yolo_labels] 경고: 이미지를 로드할 수 없음: {img_path}")
                error_count += 1
                continue

            img_h, img_w = img.shape[:2]

            # JSON 파일 로드
            json_path = os.path.join(json_dir, json_files[base_name])
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[create_yolo_labels] 경고: JSON 파싱 오류: {json_path}")
                error_count += 1
                continue

            # 새로운 JSON 형식에 맞게 처리
            if "row" not in json_data:
                print(f"[create_yolo_labels] 경고: JSON에 'row' 키가 없음: {json_path}")
                error_count += 1
                continue

            # YOLO 형식 라벨 생성
            yolo_lines = []

            for obj in json_data["row"]:
                try:
                    obj_type = obj["attributes2"]

                    # 클래스 ID 결정
                    class_id = 16  # 기본값: "기타"
                    if obj_type in class_mapping:
                        class_id = class_mapping[obj_type]

                    # 좌표 처리 (points1~points4에서 바운딩 박스 추출)
                    points = []
                    for i in range(1, 5):
                        key = f"points{i}"
                        if key in obj and obj[key]:
                            try:
                                x, y = map(float, obj[key].split(","))
                                points.append((x, y))
                            except (ValueError, AttributeError):
                                continue

                    # 4개의 점이 없다면 계속
                    if len(points) != 4:
                        print(f"[경고] 객체 좌표가 부족합니다: {json_path}")
                        continue

                    # 바운딩 박스 계산 (min_x, min_y, max_x, max_y)
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    # 좌표 검증 및 보정
                    if min_x >= max_x or min_y >= max_y:
                        continue  # 잘못된 바운딩 박스 건너뛰기

                    # 좌표 정규화 및 [0,1] 범위로 클리핑
                    x_center = normalize_and_clip((min_x + max_x) / 2, img_w)
                    y_center = normalize_and_clip((min_y + max_y) / 2, img_h)
                    box_width = normalize_and_clip(max_x - min_x, img_w)
                    box_height = normalize_and_clip(max_y - min_y, img_h)

                    # 유효성 검증 (0~1 범위)
                    if not (
                        0 <= x_center <= 1
                        and 0 <= y_center <= 1
                        and 0 < box_width <= 1
                        and 0 < box_height <= 1
                    ):
                        print(
                            f"[경고] 유효하지 않은 바운딩 박스: {json_path}, 좌표: {x_center}, {y_center}, 너비: {box_width}, 높이: {box_height}"
                            f"이미지 크기: {img_w}, {img_h}"
                            f"좌표 범위: {min_x}, {max_x}, {min_y}, {max_y}"
                        )
                        continue

                    # YOLO 형식 라인 생성 및 추가
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                    yolo_lines.append(yolo_line)

                    # 클래스 카운트 증가
                    class_counts[class_id] += 1

                except Exception as e:
                    print(f"[create_yolo_labels] 객체 처리 중 오류: {str(e)}")
                    continue

            # 라벨 파일 저장
            with open(out_file, "w") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines))

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"[create_yolo_labels] 진행 상황: {processed_count}개 처리 중...")

        except Exception as e:
            print(f"[create_yolo_labels] 예외 발생 ({base_name}): {str(e)}")
            error_count += 1
            continue

    print(
        f"[create_yolo_labels] 처리 완료: {processed_count}개 처리, {skipped_count}개 생략, {error_count}개 오류"
    )

    # 클래스별 통계 출력
    print(f"[create_yolo_labels] 클래스별 객체 수:")
    for class_id, count in class_counts.items():
        if count > 0:
            class_name = next(
                (name for name, cid in class_mapping.items() if cid == class_id),
                f"클래스 {class_id}",
            )
            print(f"  - {class_name}: {count}개")

    if processed_count == 0:
        print(
            "[create_yolo_labels] 주의: 아무 파일도 처리되지 않았습니다. 경로와 파일 형식을 확인하세요."
        )

    return processed_count
