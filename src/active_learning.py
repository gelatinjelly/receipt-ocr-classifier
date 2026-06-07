import json
import os
import numpy as np
from classifier import FewShotClassifier

FEEDBACK_PATH = "/home/kjs/receipt-classifier/models/prototypes/feedback.json"

class ActiveLearner:
    def __init__(self, classifier: FewShotClassifier):
        self.clf = classifier
        self.feedback_log = []
        self.load_feedback()

    def predict_and_ask(self, item_name, amount):
        """분류 후 신뢰도 낮으면 사용자에게 물어보기"""
        category, score = self.clf.classify(item_name)

        print(f"\n항목: {item_name} ({amount:,}원)")

        if category == "미분류" or score < 0.6:
            print(f"→ 자동 분류 실패 (유사도: {score:.2f})")
            category = self._ask_user(item_name)
        else:
            print(f"→ 자동 분류: {category} (유사도: {score:.2f})")
            confirm = input("  맞나요? (Enter=yes / 다른 카테고리 입력): ").strip()
            if confirm:
                category = self._ask_user(item_name, suggest=confirm)

        return category

    def _ask_user(self, item_name, suggest=None):
        """사용자에게 카테고리 입력받기"""
        categories = list(self.clf.prototypes.keys())

        print(f"  카테고리 목록: {', '.join(categories)}")
        if suggest:
            print(f"  입력하신 카테고리: {suggest}")

        while True:
            user_input = suggest if suggest else input("  카테고리 입력 (새 카테고리도 가능): ").strip()
            suggest = None  # 한 번만 사용

            if not user_input:
                continue

            # 새 카테고리면 예시 추가
            if user_input not in self.clf.prototypes:
                print(f"  새 카테고리 '{user_input}' 생성!")
                self.clf.add_category(user_input, [item_name])
            else:
                # 기존 카테고리 프로토타입 업데이트
                self._update_prototype(user_input, item_name)

            # 피드백 저장
            self.feedback_log.append({
                'item': item_name,
                'category': user_input
            })
            self.save_feedback()
            return user_input

    def _update_prototype(self, category, item_name):
        """피드백으로 프로토타입 벡터 업데이트"""
        new_embedding = self.clf.model.encode([item_name])
        existing = np.array(self.clf.prototypes[category])

        # 기존 벡터에 새 벡터 추가
        updated = np.vstack([existing, new_embedding])
        self.clf.prototypes[category] = updated.tolist()
        self.clf.examples[category].append(item_name)
        self.clf.save()
        print(f"  '{category}' 프로토타입 업데이트 완료!")

    def process_receipt(self, items):
        """영수증 항목 전체 처리"""
        results = []
        for item in items:
            category = self.predict_and_ask(item['name'], item['amount'])
            results.append({**item, 'category': category})
        return results

    def save_feedback(self):
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        with open(FEEDBACK_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_log, f, ensure_ascii=False, indent=2)

    def load_feedback(self):
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
                self.feedback_log = json.load(f)

if __name__ == "__main__":
    import sys
    sys.path.append('/home/kjs/receipt-classifier/src')

    clf = FewShotClassifier()
    learner = ActiveLearner(clf)

    # 테스트 항목
    test_items = [
        {'name': '아메리카노', 'amount': 4500},
        {'name': '맥북 충전기', 'amount': 35000},
        {'name': '삼겹살 2인분', 'amount': 28000},
        {'name': '넷플릭스', 'amount': 17000},
        {'name': '헬스장 월정액', 'amount': 80000},
    ]

    print("=== 능동 학습 루프 테스트 ===")
    results = learner.process_receipt(test_items)

    print("\n=== 최종 분류 결과 ===")
    for r in results:
        print(f"{r['name']:20s} → {r['category']:10s} | {r['amount']:>10,}원")
