import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import re
import easyocr
import torch
import torch.nn as nn
import sys
import os
import io
import lmdb
from PIL import Image, ImageDraw, ImageFont

sys.path.append('/home/kjs/deep-text-recognition-benchmark')
from model import Model
from utils import AttnLabelConverter
import argparse

OUTPUT_DIR = '/home/kjs/receipt-classifier/data/processed/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 학습 곡선 ──
def plot_training_curve():
    log_path = '/home/kjs/deep-text-recognition-benchmark/saved_models/receipt_ocr/log_train.txt'

    iters, train_losses, val_losses, accuracies = [], [], [], []

    with open(log_path) as f:
        content = f.read()

    pattern = r'\[(\d+)/10000\] Train loss: ([\d.]+), Valid loss: ([\d.]+).*?Current_accuracy\s*:\s*([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)

    for m in matches:
        iters.append(int(m[0]))
        train_losses.append(float(m[1]))
        val_losses.append(float(m[2]))
        accuracies.append(float(m[3]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('EasyOCR Fine-tuning on CORD Dataset', fontsize=14, fontweight='bold')

    # Loss 그래프
    ax1.plot(iters, train_losses, 'b-o', markersize=4, label='Train Loss')
    ax1.plot(iters, val_losses, 'r-o', markersize=4, label='Val Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train / Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy 그래프
    ax2.plot(iters, accuracies, 'g-o', markersize=4, label='Val Accuracy')
    ax2.axhline(y=93.4, color='orange', linestyle='--', label='Test Accuracy (93.4%)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/training_curve.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'학습 곡선 저장: {path}')
    plt.close()

# ── 2. 전/후 비교 이미지 ──
def load_finetuned_model():
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
        map_location=device, weights_only=False
    ))
    model.eval()
    return model, converter, opt, device

def predict_finetuned(model, converter, opt, device, image):
    img = image.convert('L').resize((opt.imgW, opt.imgH), Image.LANCZOS)
    img_tensor = torch.FloatTensor(np.array(img)).unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        batch_size = 1
        length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        preds = model(img_tensor, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        return preds_str[0].split('[s]')[0].strip().lower()

def plot_comparison():
    print("모델 로드 중...")
    ft_model, converter, opt, device = load_finetuned_model()
    easy_reader = easyocr.Reader(['en'], gpu=True)

    # test LMDB에서 샘플 20개 뽑기
    env = lmdb.open('/home/kjs/receipt-classifier/data/lmdb/test', readonly=True, lock=False)
    samples = []
    with env.begin() as txn:
        n = int(txn.get(b'num-samples').decode())
        for idx in range(1, n + 1):
            if len(samples) >= 20:
                break
            img_buf = txn.get(f'image-{idx:09d}'.encode())
            label = txn.get(f'label-{idx:09d}'.encode())
            if img_buf and label:
                label_str = label.decode().strip().lower()
                if all(c in '0123456789abcdefghijklmnopqrstuvwxyz' for c in label_str):
                    samples.append((img_buf, label_str))

    # 비교 그리드 생성
    fig, axes = plt.subplots(5, 4, figsize=(16, 12))
    fig.suptitle('OCR Comparison: EasyOCR vs Fine-tuned Model', fontsize=14, fontweight='bold')

    for i, (img_buf, gt) in enumerate(samples):
        row, col = i // 4, i % 4
        ax = axes[row][col]

        image = Image.open(io.BytesIO(img_buf)).convert('RGB')

        # EasyOCR 예측
        try:
            results = easy_reader.readtext(np.array(image), detail=0)
            easy_pred = results[0].strip().lower() if results else ''
        except:
            easy_pred = ''

        # 파인튜닝 예측
        ft_pred = predict_finetuned(ft_model, converter, opt, device, image)

        ax.imshow(image)
        ax.axis('off')

        easy_color = 'green' if easy_pred == gt else 'red'
        ft_color = 'green' if ft_pred == gt else 'red'

        title = f'GT: {gt}\nBase: {easy_pred} | FT: {ft_pred}'
        ax.set_title(title, fontsize=7,
                     color='black' if easy_pred == gt and ft_pred == gt else 'darkred')

        # 테두리 색으로 파인튜닝 결과 표시
        for spine in ax.spines.values():
            spine.set_edgecolor(ft_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'비교 이미지 저장: {path}')
    plt.close()

# ── 3. 최종 결과 바 차트 ──
def plot_final_results():
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['EasyOCR\n기본 모델', '파인튜닝 모델\n(CORD)']
    accuracies = [53.033, 93.388]
    colors = ['#E74C3C', '#27AE60']

    bars = ax.bar(models, accuracies, color=colors, width=0.4, edgecolor='white')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylim([0, 110])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('EasyOCR Fine-tuning 전/후 정확도 비교\n(동일 샘플 1,467개)', fontsize=13, fontweight='bold')
    ax.axhline(y=53.033, color='#E74C3C', linestyle='--', alpha=0.4)
    ax.axhline(y=93.388, color='#27AE60', linestyle='--', alpha=0.4)
    ax.grid(True, axis='y', alpha=0.3)

    ax.annotate('', xy=(1, 93.388), xytext=(1, 53.033),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(1.22, 73, '+40.4%p', fontsize=13, fontweight='bold', color='black')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/final_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'최종 비교 저장: {path}')
    plt.close()

if __name__ == "__main__":
    pip_check = os.system('pip show matplotlib > /dev/null 2>&1')
    if pip_check != 0:
        os.system('pip install matplotlib')

    print("=== 시각화 생성 중 ===")
    plot_training_curve()
    plot_comparison()
    plot_final_results()
    print("\n완료! 저장 위치:")
    print(f"  {OUTPUT_DIR}/training_curve.png")
    print(f"  {OUTPUT_DIR}/comparison.png")
    print(f"  {OUTPUT_DIR}/final_comparison.png")
