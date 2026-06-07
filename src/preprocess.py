import cv2
import numpy as np
from PIL import Image

def preprocess(image: Image.Image) -> Image.Image:
    img = np.array(image)

    # 1. 그레이스케일
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. CLAHE (조명 불균일 보정)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. 적응형 이진화 (배경 복잡해도 잘 작동)
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # 4. 노이즈 제거
    denoised = cv2.medianBlur(binary, 3)

    return Image.fromarray(denoised)

if __name__ == "__main__":
    import easyocr
    from datasets import load_dataset

    dataset = load_dataset('naver-clova-ix/cord-v2')
    sample = dataset['train'][0]
    original = sample['image']

    processed = preprocess(original)
    processed.save('/home/kjs/receipt-classifier/data/raw/sample_0_processed.png')
    print("저장 완료!")

    reader = easyocr.Reader(['ko', 'en'], gpu=True)

    print("\n=== 보정 후 OCR ===")
    after = reader.readtext(np.array(processed))
    for (_, text, conf) in after:
        if conf > 0.3:
            print(f"{text:30s} | {conf:.2f}")
