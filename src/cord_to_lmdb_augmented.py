import os
import json
import lmdb
import numpy as np
from PIL import Image
from datasets import load_dataset
import io
import cv2

def augment_image(image):
    """데이터 증강 적용"""
    img = np.array(image)
    
    # 랜덤하게 증강 선택
    aug_type = np.random.randint(0, 4)
    
    if aug_type == 0:
        # 밝기 조절
        factor = np.random.uniform(0.6, 1.4)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    
    elif aug_type == 1:
        # 가우시안 노이즈
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    
    elif aug_type == 2:
        # 블러
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    elif aug_type == 3:
        # 회전 (±10도)
        angle = np.random.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(img)

def word_box_to_crops_augmented(image, ground_truth, augment=True):
    """CORD GT에서 단어별 이미지 크롭 + 증강"""
    crops = []
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        valid_line = gt.get('valid_line', [])

        for line in valid_line:
            for word in line.get('words', []):
                text = word.get('text', '').strip()
                quad = word.get('quad', {})

                if not text or len(text) < 1:
                    continue

                try:
                    x_coords = [quad['x1'], quad['x2'], quad['x3'], quad['x4']]
                    y_coords = [quad['y1'], quad['y2'], quad['y3'], quad['y4']]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)

                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue

                    crop = image.crop((x1, y1, x2, y2))
                    crop = crop.resize((100, 32), Image.LANCZOS)

                    # 원본 저장
                    crops.append((crop, text))

                    # 증강본 추가 (train만)
                    if augment:
                        aug = augment_image(crop)
                        crops.append((aug, text))

                except:
                    continue
    except:
        pass

    return crops

def create_lmdb_augmented(dataset_split, output_path, augment=True):
    os.makedirs(output_path, exist_ok=True)

    all_crops = []
    print(f"데이터 준비 중... (증강: {augment})")

    for i, sample in enumerate(dataset_split):
        crops = word_box_to_crops_augmented(
            sample['image'], sample['ground_truth'], augment=augment
        )
        all_crops.extend(crops)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}번째 이미지 처리 중... (누적 {len(all_crops)}개)")

    print(f"총 {len(all_crops)}개 단어 추출 완료")

    map_size = max(len(all_crops) * 1000 * 100, 1073741824)
    env = lmdb.open(output_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, (crop, text) in enumerate(all_crops):
            img_buf = io.BytesIO()
            crop.save(img_buf, format='PNG')
            txn.put(f'image-{idx+1:09d}'.encode(), img_buf.getvalue())
            txn.put(f'label-{idx+1:09d}'.encode(), text.encode())
        txn.put(b'num-samples', str(len(all_crops)).encode())

    print(f"LMDB 저장 완료: {output_path}")
    return len(all_crops)

if __name__ == "__main__":
    print("CORD 데이터셋 로드 중...")
    dataset = load_dataset('naver-clova-ix/cord-v2')

    # Train만 증강 적용
    n_train = create_lmdb_augmented(
        dataset['train'],
        '/home/kjs/receipt-classifier/data/lmdb/train_augmented',
        augment=True
    )

    print(f"\n완료! Train (증강): {n_train}개")
    print(f"기존 Train: 19,367개 → 증강 후: {n_train}개")
