import os
import json
import lmdb
import numpy as np
from PIL import Image
from datasets import load_dataset
import io

def word_box_to_crops(image, ground_truth):
    """CORD GT에서 단어별 이미지 크롭 + 텍스트 추출"""
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

                # 좌표 추출
                try:
                    x_coords = [quad['x1'], quad['x2'], quad['x3'], quad['x4']]
                    y_coords = [quad['y1'], quad['y2'], quad['y3'], quad['y4']]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)

                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue

                    crop = image.crop((x1, y1, x2, y2))
                    crop = crop.resize((100, 32), Image.LANCZOS)
                    crops.append((crop, text))
                except:
                    continue
    except:
        pass

    return crops

def create_lmdb(dataset_split, output_path, max_samples=None):
    """CORD 데이터셋 → LMDB 변환"""
    os.makedirs(output_path, exist_ok=True)

    # 전체 크기 파악
    all_crops = []
    print(f"데이터 준비 중...")

    for i, sample in enumerate(dataset_split):
        if max_samples and i >= max_samples:
            break
        crops = word_box_to_crops(sample['image'], sample['ground_truth'])
        all_crops.extend(crops)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}번째 이미지 처리 중... (누적 {len(all_crops)}개 단어)")

    print(f"총 {len(all_crops)}개 단어 추출 완료")

    # LMDB 저장
    map_size = len(all_crops) * 1000 * 100
    env = lmdb.open(output_path, map_size=max(map_size, 1073741824))

    with env.begin(write=True) as txn:
        for idx, (crop, text) in enumerate(all_crops):
            # 이미지 저장
            img_buf = io.BytesIO()
            crop.save(img_buf, format='PNG')
            img_key = f'image-{idx+1:09d}'.encode()
            txn.put(img_key, img_buf.getvalue())

            # 라벨 저장
            label_key = f'label-{idx+1:09d}'.encode()
            txn.put(label_key, text.encode())

        # 전체 개수 저장
        txn.put(b'num-samples', str(len(all_crops)).encode())

    print(f"LMDB 저장 완료: {output_path}")
    return len(all_crops)

if __name__ == "__main__":
    print("CORD 데이터셋 로드 중...")
    dataset = load_dataset('naver-clova-ix/cord-v2')

    # Train
    n_train = create_lmdb(
        dataset['train'],
        '/home/kjs/receipt-classifier/data/lmdb/train'
    )

    # Validation
    n_val = create_lmdb(
        dataset['validation'],
        '/home/kjs/receipt-classifier/data/lmdb/val'
    )

    print(f"\n완료! Train: {n_train}개 / Val: {n_val}개")
