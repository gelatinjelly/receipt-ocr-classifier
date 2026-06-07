import json
import os
from datetime import datetime
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DATA_PATH = "/home/kjs/receipt-classifier/data/processed/expenses.json"

class ExpenseAnalyzer:
    def __init__(self):
        self.expenses = []
        self.load()

    def add_items(self, items, date=None):
        """분류된 항목 추가"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        for item in items:
            self.expenses.append({
                'name': item['name'],
                'amount': item['amount'],
                'category': item['category'],
                'date': date
            })
        self.save()
        print(f"{len(items)}개 항목 저장 완료!")

    def summary(self):
        """카테고리별 소비 요약"""
        totals = defaultdict(int)
        counts = defaultdict(int)
        for e in self.expenses:
            totals[e['category']] += e['amount']
            counts[e['category']] += 1

        print("\n=== 소비 요약 ===")
        total_all = sum(totals.values())
        for cat, amount in sorted(totals.items(), key=lambda x: -x[1]):
            pct = amount / total_all * 100
            print(f"{cat:10s} | {amount:>10,}원 | {pct:.1f}% | {counts[cat]}건")
        print(f"{'합계':10s} | {total_all:>10,}원")
        return totals

    def detect_anomaly(self):
        """이상 지출 탐지"""
        # 카테고리별 평균 대비 이번 지출 비교
        from collections import defaultdict
        cat_amounts = defaultdict(list)
        for e in self.expenses:
            cat_amounts[e['category']].append(e['amount'])

        print("\n=== 이상 지출 탐지 ===")
        anomalies = []
        for cat, amounts in cat_amounts.items():
            if len(amounts) < 2:
                continue
            avg = sum(amounts[:-1]) / len(amounts[:-1])
            latest = amounts[-1]
            ratio = latest / avg if avg > 0 else 0
            if ratio > 1.3:
                msg = f"'{cat}' 최근 지출이 평균보다 {(ratio-1)*100:.0f}% 높아요!"
                print(f"⚠️  {msg}")
                anomalies.append(msg)
        if not anomalies:
            print("이상 지출 없음")
        return anomalies

    def visualize(self):
        """소비 패턴 시각화"""
        if not self.expenses:
            print("데이터 없음")
            return

        totals = defaultdict(int)
        for e in self.expenses:
            totals[e['category']] += e['amount']

        categories = list(totals.keys())
        amounts = list(totals.values())

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("카테고리별 지출", "지출 비율"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )

        # 바 차트
        fig.add_trace(
            go.Bar(
                x=categories,
                y=amounts,
                marker_color='steelblue',
                text=[f"{a:,}원" for a in amounts],
                textposition='auto'
            ),
            row=1, col=1
        )

        # 파이 차트
        fig.add_trace(
            go.Pie(
                labels=categories,
                values=amounts,
                hole=0.3
            ),
            row=1, col=2
        )

        fig.update_layout(
            title="소비 패턴 분석",
            showlegend=True,
            height=500
        )

        output_path = "/home/kjs/receipt-classifier/data/processed/analysis.html"
        fig.write_html(output_path)
        print(f"\n시각화 저장 완료: {output_path}")

    def save(self):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.expenses, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                self.expenses = json.load(f)

if __name__ == "__main__":
    analyzer = ExpenseAnalyzer()

    # 테스트 데이터
    test_results = [
        {'name': '아메리카노', 'amount': 4500, 'category': '카페'},
        {'name': '맥북 충전기', 'amount': 35000, 'category': '전자기기'},
        {'name': '삼겹살 2인분', 'amount': 28000, 'category': '식비'},
        {'name': '넷플릭스', 'amount': 17000, 'category': 'ott'},
        {'name': '헬스장 월정액', 'amount': 80000, 'category': '건강'},
        {'name': '카페라떼', 'amount': 5000, 'category': '카페'},
        {'name': '편의점 도시락', 'amount': 4800, 'category': '식비'},
        {'name': '카카오택시', 'amount': 12000, 'category': '교통'},
    ]

    analyzer.add_items(test_results)
    analyzer.summary()
    analyzer.detect_anomaly()
    analyzer.visualize()
