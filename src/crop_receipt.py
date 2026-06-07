import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

def order_points(pts):
    """꼭짓점 정렬 (좌상, 우상, 우하, 좌하)"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_receipt(image: Image.Image) -> Image.Image:
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. 블러 + 엣지 검출
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # 2. 윤곽선 찾기
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    receipt_contour = None
    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            break

    # 윤곽선 못 찾으면 원본 반환
    if receipt_contour is None:
        print("영수증 윤곽선 못 찾음 → 원본 사용")
        return image

    # 3. 원근 변환으로 영수증만 크롭
    pts = receipt_contour.reshape(4, 2).astype("float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return Image.fromarray(warped)

if __name__ == "__main__":
    import sys
    sys.path.append('/home/kjs/receipt-classifier/src')
    from preprocess import preprocess
    import easyocr

    dataset = load_dataset('naver-clova-ix/cord-v2')
    sample = dataset['train'][0]
    original = sample['image']

    # 1. 크롭 먼저
    cropped = crop_receipt(original)
    cropped.save('/home/kjs/receipt-classifier/data/raw/sample_0_cropped.png')
    print(f"크롭 후 크기: {cropped.size}")

    # 2. 크롭 후 전처리
    processed = preprocess(cropped)
    processed.save('/home/kjs/receipt-classifier/data/raw/sample_0_cropped_processed.png')

    # 3. OCR
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    results = reader.readtext(np.array(processed))

    print("\n=== 크롭 + 보정 후 OCR ===")
    for (_, text, conf) in results:
        if conf > 0.5:
            print(f"{text:30s} | {conf:.2f}")
