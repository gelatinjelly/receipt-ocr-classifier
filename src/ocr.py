import easyocr
import json
import numpy as np
from datasets import load_dataset
from PIL import Image

def run_ocr(image):
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    results = reader.readtext(np.array(image))
    return results

def main():
    # CORD 샘플 로드
    print("데이터셋 로드 중...")
    dataset = load_dataset('naver-clova-ix/cord-v2')
    sample = dataset['train'][0]

    image = sample['image']
    gt = json.loads(sample['ground_truth'])

    # OCR 실행
    print("OCR 실행 중...")
    results = run_ocr(image)

    print("\n=== OCR 결과 ===")
    for (bbox, text, confidence) in results:
        print(f"텍스트: {text:30s} | 신뢰도: {confidence:.2f}")

    print("\n=== 정답 라벨 (가게명) ===")
    print(gt['gt_parse'].get('store_name', '없음'))

if __name__ == "__main__":
    main()
