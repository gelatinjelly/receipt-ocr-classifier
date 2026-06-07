import sys
import os
import lmdb
import torch
import torch.nn as nn
import numpy as np
import easyocr
from PIL import Image
import io

sys.path.append('/home/kjs/deep-text-recognition-benchmark')

from model import Model
from utils import CTCLabelConverter, AttnLabelConverter
import argparse

def load_finetuned_model():
    """파인튜닝된 모델 로드"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--Transformation', default='TPS')
    parser.add_argument('--FeatureExtraction', default='ResNet')
    parser.add_argument('--SequenceModeling', default='BiLSTM')
    parser.add_argument('--Prediction', default='Attn')
    parser.add_argument('--num_fiducial', type=int, default=20)
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_max_length', type=int, default=25)
    parser.add_argument('--character', default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--sensitive', action='store_true')
    parser.add_argument('--PAD', action='store_true')
    opt = parser.parse_args([])

    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(opt)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(
        '/home/kjs/deep-text-recognition-benchmark/saved_models/receipt_ocr/best_accuracy.pth',
        map_location=device,
        weights_only=False
    ))
    model.eval()
    return model, converter, opt, device

def predict_finetuned(model, converter, opt, device, image):
    """파인튜닝 모델로 예측"""
    img = image.convert('L').resize((opt.imgW, opt.imgH), Image.LANCZOS)
    img_tensor = torch.FloatTensor(np.array(img)).unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        batch_size = img_tensor.size(0)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        preds = model(img_tensor, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        pred = preds_str[0].split('[s]')[0].strip().lower()
    return pred

def compare_models():
    LMDB_PATH = '/home/kjs/receipt-classifier/data/lmdb/test'

    print("모델 로드 중...")
    ft_model, converter, opt, device = load_finetuned_model()
    easy_reader = easyocr.Reader(['en'], gpu=True)

    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)

    with env.begin() as txn:
        n_samples = int(txn.get(b'num-samples').decode())

        # 샘플 로드
        samples = []
        for idx in range(1, n_samples + 1):
            img_key = f'image-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()
            img_buf = txn.get(img_key)
            label = txn.get(label_key)
            if img_buf and label:
                label_str = label.decode().strip().lower()
                # 파인튜닝 모델 character set에 있는 것만
                if all(c in '0123456789abcdefghijklmnopqrstuvwxyz' for c in label_str):
                    samples.append((img_buf, label_str))

    print(f"공통 평가 샘플: {len(samples)}개\n")

    easy_correct = 0
    ft_correct = 0
    total = len(samples)

    for i, (img_buf, gt) in enumerate(samples):
        image = Image.open(io.BytesIO(img_buf)).convert('RGB')

        # EasyOCR 예측
        try:
            results = easy_reader.readtext(np.array(image), detail=0)
            easy_pred = results[0].strip().lower() if results else ''
        except:
            easy_pred = ''

        # 파인튜닝 모델 예측
        ft_pred = predict_finetuned(ft_model, converter, opt, device, image)

        if easy_pred == gt:
            easy_correct += 1
        if ft_pred == gt:
            ft_correct += 1

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{total}] EasyOCR: {easy_correct/(i+1)*100:.1f}% | 파인튜닝: {ft_correct/(i+1)*100:.1f}%")

    print(f"\n{'='*50}")
    print(f"{'모델':<20} {'정확도':>10}")
    print(f"{'='*50}")
    print(f"{'EasyOCR 기본':<20} {easy_correct/total*100:>9.3f}%")
    print(f"{'파인튜닝 모델':<20} {ft_correct/total*100:>9.3f}%")
    print(f"{'향상폭':<20} {(ft_correct-easy_correct)/total*100:>+9.3f}%p")
    print(f"{'='*50}")

if __name__ == "__main__":
    compare_models()
