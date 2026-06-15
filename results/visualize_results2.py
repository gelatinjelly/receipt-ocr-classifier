import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import re
import os

# 한글 폰트 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/home/kjs/receipt-classifier/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 전체 실험 결과 바 차트 ──
def plot_all_results():
    labels = [
        'EasyOCR\n기본 모델',
        'TPS-ResNet\nBiLSTM-Attn\n(파인튜닝)',
        'None-ResNet\nBiLSTM-CTC\n(구조 변경)',
        'TPS-ResNet\nBiLSTM-Attn\n(증강 학습)',
    ]
    accuracies = [53.0, 93.4, 92.8, 93.8]
    colors = ['#E74C3C', '#27AE60', '#2980B9', '#8E44AD']

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, accuracies, color=colors, width=0.5, edgecolor='white')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{acc:.1f}%', ha='center', va='bottom',
                fontsize=13, fontweight='bold')

    ax.set_ylim([0, 110])
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('전체 실험 결과 비교\n(동일 Test 샘플 1,467개)', fontsize=14, fontweight='bold')
    ax.axhline(y=53.0, color='#E74C3C', linestyle='--', alpha=0.3)
    ax.grid(True, axis='y', alpha=0.3)

    # 향상폭 표시
    for i in range(1, 4):
        diff = accuracies[i] - accuracies[0]
        ax.annotate(f'+{diff:.1f}%p',
                   xy=(i, accuracies[i]),
                   xytext=(i, accuracies[i] + 5),
                   ha='center', fontsize=10,
                   color=colors[i], fontweight='bold')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/all_results.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'전체 결과 저장: {path}')
    plt.close()

# ── 2. 학습 곡선 3개 비교 ──
def parse_log(log_path):
    iters, accuracies = [], []
    with open(log_path) as f:
        content = f.read()
    pattern = r'\[(\d+)/10000\].*?Current_accuracy\s*:\s*([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    for m in matches:
        iters.append(int(m[0]))
        accuracies.append(float(m[1]))
    return iters, accuracies

def plot_training_curves():
    logs = {
        'TPS-Attn (파인튜닝)': '/home/kjs/deep-text-recognition-benchmark/saved_models/receipt_ocr/log_train.txt',
        'CTC (구조 변경)': '/home/kjs/deep-text-recognition-benchmark/saved_models/receipt_ocr_ctc/log_train.txt',
        'TPS-Attn (증강)': '/home/kjs/deep-text-recognition-benchmark/saved_models/receipt_ocr_augmented/log_train.txt',
    }
    colors = ['#27AE60', '#2980B9', '#8E44AD']

    fig, ax = plt.subplots(figsize=(12, 6))

    for (label, path), color in zip(logs.items(), colors):
        iters, accs = parse_log(path)
        ax.plot(iters, accs, 'o-', color=color, label=label, markersize=5)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('모델별 학습 곡선 비교', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/training_curves_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'학습 곡선 비교 저장: {path}')
    plt.close()

# ── 3. 테스트별 요약 표 ──
def plot_summary_table():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    data = [
        ['기준', 'EasyOCR 기본', '-', '-', '53.0%', '-'],
        ['테스트 1', 'TPS-ResNet-BiLSTM-Attn', 'CORD v2\n19,367개', '없음', '93.4%', '+40.4%p'],
        ['테스트 2', 'None-ResNet-BiLSTM-CTC', 'CORD v2\n19,367개', '없음', '92.8%', '+39.8%p'],
        ['테스트 3', 'TPS-ResNet-BiLSTM-Attn', 'CORD v2\n38,734개', '회전/노이즈\n밝기/블러', '93.8%', '+40.8%p'],
    ]

    columns = ['구분', '모델 구조', '학습 데이터', '증강', 'Test Acc', '향상폭']

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # 헤더 색상
    for j in range(len(columns)):
        table[0, j].set_facecolor('#1E2761')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 행 색상
    row_colors = ['#F4F6FB', 'white', '#EAF7EF', '#EBF5FB', '#F5EEF8']
    for i in range(1, len(data)+1):
        for j in range(len(columns)):
            table[i, j].set_facecolor(row_colors[i])

    # 향상폭 색상
    for i in range(1, len(data)+1):
        table[i, 5].set_text_props(color='#27AE60', fontweight='bold')

    plt.title('실험 결과 요약', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/summary_table.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'요약 표 저장: {path}')
    plt.close()

if __name__ == "__main__":
    print("=== 시각화 생성 중 ===")
    plot_all_results()
    plot_training_curves()
    plot_summary_table()
    print("\n완료!")
