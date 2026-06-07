import easyocr
import lmdb
import numpy as np
from PIL import Image
import io

def eval_easyocr_on_lmdb(lmdb_path):
    """기본 EasyOCR로 LMDB test셋 평가"""
    reader = easyocr.Reader(['en'], gpu=True)
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    with env.begin() as txn:
        n_samples = int(txn.get(b'num-samples').decode())
        print(f"총 {n_samples}개 샘플 평가 중...")
        
        correct = 0
        total = 0
        
        for idx in range(1, n_samples + 1):
            # 이미지 로드
            img_key = f'image-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()
            
            img_buf = txn.get(img_key)
            label = txn.get(label_key).decode()
            
            if img_buf is None:
                continue
            
            # EasyOCR 인식
            image = Image.open(io.BytesIO(img_buf)).convert('RGB')
            img_np = np.array(image)
            
            try:
                results = reader.readtext(img_np, detail=0)
                pred = results[0].strip().lower() if results else ''
            except:
                pred = ''
            
            gt = label.strip().lower()
            
            if pred == gt:
                correct += 1
            total += 1
            
            if total % 200 == 0:
                print(f"  {total}/{n_samples} | 현재 정확도: {correct/total*100:.2f}%")
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n=== EasyOCR 기본 모델 정확도: {accuracy:.3f}% ===")
    return accuracy

if __name__ == "__main__":
    eval_easyocr_on_lmdb('/home/kjs/receipt-classifier/data/lmdb/test')
