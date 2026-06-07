import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "jhgan/ko-sroberta-multitask"
PROTOTYPES_PATH = "/home/kjs/receipt-classifier/models/prototypes/prototypes.json"

class FewShotClassifier:
    def __init__(self, threshold=0.4):
        print("모델 로딩 중...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.threshold = threshold
        self.prototypes = {}  # {카테고리: [임베딩 리스트]}
        self.examples = {}    # {카테고리: [예시 텍스트]}
        self.load()

    def add_category(self, category, examples):
        """카테고리 + 예시 추가"""
        embeddings = self.model.encode(examples)
        self.prototypes[category] = embeddings.tolist()
        self.examples[category] = examples
        self.save()
        print(f"카테고리 '{category}' 추가 완료 ({len(examples)}개 예시)")

    def classify(self, item_name):
        """항목명 → 카테고리 분류"""
        if not self.prototypes:
            return "미분류", 0.0

        embedding = self.model.encode([item_name])

        best_category = "미분류"
        best_score = 0.0

        for category, proto_embeddings in self.prototypes.items():
            sims = cosine_similarity(embedding, proto_embeddings)
            score = float(np.max(sims))
            if score > best_score:
                best_score = score
                best_category = category

        if best_score < self.threshold:
            return "미분류", best_score

        return best_category, best_score

    def save(self):
        os.makedirs(os.path.dirname(PROTOTYPES_PATH), exist_ok=True)
        with open(PROTOTYPES_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'prototypes': self.prototypes,
                'examples': self.examples
            }, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(PROTOTYPES_PATH):
            with open(PROTOTYPES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.prototypes = data.get('prototypes', {})
                self.examples = data.get('examples', {})
            print(f"기존 프로토타입 로드: {list(self.prototypes.keys())}")

if __name__ == "__main__":
    clf = FewShotClassifier()

    # 카테고리 정의 (한국어 기준)
    clf.add_category("식비", [
        "삼겹살", "치킨", "피자", "햄버거", "김치찌개",
        "된장찌개", "비빔밥", "냉면", "짜장면", "초밥"
    ])
    clf.add_category("카페", [
        "아메리카노", "카페라떼", "스타벅스", "카페", "커피",
        "에이드", "스무디", "케이크", "마카롱", "음료"
    ])
    clf.add_category("교통", [
        "택시", "버스", "지하철", "주유", "카카오택시",
        "우버", "기차", "KTX", "고속버스", "톨게이트"
    ])
    clf.add_category("쇼핑", [
        "올리브영", "다이소", "쿠팡", "옷", "신발",
        "화장품", "세제", "샴푸", "마스크팩", "향수"
    ])
    clf.add_category("건강", [
        "헬스장", "약국", "비타민", "단백질", "보충제",
        "병원", "치과", "약", "마스크", "체육관"
    ])

    # 테스트
    test_items = [
        "아메리카노", "삼겹살 2인분", "카카오택시",
        "올리브영 비타민", "헬스장 월정액", "맥북 충전기"
    ]

    print("\n=== 분류 테스트 ===")
    for item in test_items:
        category, score = clf.classify(item)
        print(f"{item:20s} → {category:10s} (유사도: {score:.2f})")
