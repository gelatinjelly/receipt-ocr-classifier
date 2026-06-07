import re
import easyocr
import numpy as np
from PIL import Image
from datasets import load_dataset
from preprocess import preprocess

def ocr_to_lines(ocr_results, conf_threshold=0.3, y_gap=10):
    tokens = []
    for (bbox, text, conf) in ocr_results:
        if conf > conf_threshold:
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            tokens.append({'x': x_center, 'y': y_center, 'text': text.strip()})

    if not tokens:
        return []

    tokens.sort(key=lambda t: t['y'])

    groups = []
    current = [tokens[0]]
    for token in tokens[1:]:
        if abs(token['y'] - current[-1]['y']) <= y_gap:
            current.append(token)
        else:
            groups.append(current)
            current = [token]
    groups.append(current)

    lines = []
    for group in groups:
        group.sort(key=lambda t: t['x'])
        line_text = ' '.join([t['text'] for t in group])
        lines.append(line_text)

    return lines

def parse_lines(lines):
    items = []
    skip_keywords = ['sub-total', 'subtotal', 'sub total', 'service',
                     'tax', 'rounding', 'grand', 'total', 'discount',
                     'pb1', 'ppn', 'frj', 'founding', 'receipt']

    amount_pattern = re.compile(r'\b\d{1,3}(?:,\d{3})+\b|\b\d{4,7}\b')

    for line in lines:
        if any(kw in line.lower() for kw in skip_keywords):
            continue

        amounts = amount_pattern.findall(line)
        if not amounts:
            continue

        amount_str = amounts[-1].replace(',', '')
        try:
            amount = int(amount_str)
            if not (1000 <= amount <= 9999999):
                continue

            name = amount_pattern.sub('', line).strip()
            name = re.sub(r'^\d+\s*[xX\.]\s*', '', name).strip()
            name = re.sub(r'\s+', ' ', name).strip()

            if len(name) >= 2:
                items.append({'name': name, 'amount': amount})
        except:
            continue

    return items

def process_image(image: Image.Image):
    """전체 파이프라인: 이미지 → 파싱 결과"""
    processed = preprocess(image)
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    results = reader.readtext(np.array(processed))
    lines = ocr_to_lines(results)
    items = parse_lines(lines)
    return items

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(__file__))

    dataset = load_dataset('naver-clova-ix/cord-v2')

    # 여러 샘플 테스트
    for idx in [0, 1, 3]:
        print(f"\n{'='*50}")
        print(f"샘플 {idx}")
        print('='*50)
        sample = dataset['train'][idx]
        items = process_image(sample['image'])

        if items:
            total = 0
            for item in items:
                print(f"항목: {item['name']:35s} | 금액: {item['amount']:>10,}")
                total += item['amount']
            print(f"합계: {total:,} | 총 {len(items)}개")
        else:
            print("파싱 실패")
