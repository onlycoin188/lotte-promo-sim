"""
합성(가상) 홈쇼핑 방송 데이터 생성기.

실제 롯데홈쇼핑 데이터는 본선에서 제공되므로, 예선 데모는 '정답을 알고 있는' 가상 데이터로 만든다.
정답(진짜 프로모션 효과)을 알고 있기 때문에, 나중에 모델이 그 효과를 얼마나 잘 찾아내는지 검증할 수 있다.

데이터에 일부러 넣은 현실적인 특징
- 카테고리마다 반응하는 프로모션이 다르다 (가전: 무이자/카드할인, 뷰티·식품: 사은품)
- 할인을 키울수록 효과가 줄어든다 (한계효용 체감)
- 같은 상품에 프로모션을 자주 걸면 효과가 떨어진다 (프로모션 피로도)
- 사은품·할인으로 산 고객은 반품을 더 많이 한다 (충동구매)
- 황금 시간대·고가 상품에 프로모션이 더 자주 걸린다 (교란 요인: 단순 비교하면 효과가 과대평가됨)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42

# ── 프로모션 선택지 ─────────────────────────────────────────────
CARD_DISC_LEVELS = [0, 5, 7, 10, 15]        # 카드 청구할인 (%)
CARD_SHARE_LEVELS = [0.0, 0.3, 0.5]          # 카드사가 부담하는 할인 비율
INSTALLMENT_LEVELS = [0, 3, 6, 10, 12]       # 무이자 할부 (개월)
GIFT_LEVELS = [0, 3, 5, 8, 12]               # 사은품 원가 (판매가 대비 %)

INSTALLMENT_FEE_PER_MONTH = 0.0035           # 무이자 할부 수수료 (판매가 대비, 개월당) — 가정값
RETURN_HANDLING_COST = 5000                  # 반품 1건 처리비 (원) — 가정값

# ── 카테고리별 특성 (진짜 효과 = 정답) ─────────────────────────────
CATEGORIES = {
    #        가격 범위(원)          마진율 범위     기본반품률  할인효과  무이자효과  사은품효과
    "가전": dict(price=(300_000, 2_500_000), margin=(0.15, 0.22), ret=0.06, disc=0.60, inst=0.35, gift=0.10),
    "뷰티": dict(price=(39_000, 120_000),    margin=(0.35, 0.50), ret=0.04, disc=0.35, inst=0.03, gift=0.45),
    "패션": dict(price=(59_000, 300_000),    margin=(0.30, 0.45), ret=0.15, disc=0.40, inst=0.10, gift=0.20),
    "식품": dict(price=(29_000, 90_000),     margin=(0.25, 0.35), ret=0.02, disc=0.30, inst=0.00, gift=0.40),
    "리빙": dict(price=(49_000, 400_000),    margin=(0.25, 0.35), ret=0.06, disc=0.45, inst=0.20, gift=0.20),
}

PRODUCT_NAMES = {
    "가전": ["무선청소기", "공기청정기", "로봇청소기", "식기세척기", "안마의자", "의류건조기", "에어프라이어", "김치냉장고"],
    "뷰티": ["기초화장품세트", "탄력앰플", "선크림세트", "쿠션팩트", "헤어케어세트", "클렌징세트", "아이크림", "바디케어세트"],
    "패션": ["캐시미어코트", "다운패딩", "니트세트", "레깅스팩", "가죽핸드백", "운동화", "속옷세트", "스카프"],
    "식품": ["한우세트", "갈비찜", "김치", "견과류", "홍삼정", "고등어", "전복", "건강즙"],
    "리빙": ["침구세트", "프라이팬세트", "수납정리함", "매트리스토퍼", "커튼", "식기세트", "정수기필터", "욕실용품세트"],
}

# 시간대별 시청/구매 가중치 (6시 ~ 25시=새벽1시)
HOUR_FACTOR = {6: 0.5, 7: 0.7, 8: 0.9, 9: 1.1, 10: 1.2, 11: 1.1, 12: 0.9, 13: 0.9, 14: 0.85,
               15: 0.85, 16: 0.9, 17: 0.9, 18: 1.0, 19: 1.15, 20: 1.3, 21: 1.35, 22: 1.25,
               23: 1.0, 24: 0.7, 25: 0.5}
# 카테고리별로 잘 팔리는 시간대 보정
CATEGORY_HOUR_BONUS = {
    "가전": {20: 1.15, 21: 1.2, 22: 1.15},
    "식품": {9: 1.15, 10: 1.2, 11: 1.15, 17: 1.1},
    "뷰티": {22: 1.15, 23: 1.2},
}
WEEKDAY_FACTOR = [0.95, 0.95, 0.97, 0.97, 1.0, 1.1, 1.08]  # 월~일


def make_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for cat, spec in CATEGORIES.items():
        for i, name in enumerate(PRODUCT_NAMES[cat]):
            lo, hi = spec["price"]
            price = float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), -3))
            # 비쌀수록 기본 주문 수는 적다
            base_orders = 15_000_000 / price ** 0.85 * rng.uniform(0.7, 1.3)
            rows.append(dict(
                product_id=f"P{len(rows) + 1:03d}",
                product_name=name,
                category=cat,
                price=price,
                margin_rate=round(rng.uniform(*spec["margin"]), 3),
                base_orders=base_orders,
            ))
    return pd.DataFrame(rows)


# ── 진짜(정답) 수요 함수 ───────────────────────────────────────────
def true_log_orders(base_orders, category, hour, weekday, card_disc, installment, gift_pct, recent_promo_cnt):
    """정답 수요 함수: 노이즈 없는 '기대 주문 수'의 로그값. (배열 입력 가능)"""
    category = np.asarray(category)
    hour = np.asarray(hour)
    disc_e = np.vectorize(lambda c: CATEGORIES[c]["disc"])(category)
    inst_e = np.vectorize(lambda c: CATEGORIES[c]["inst"])(category)
    gift_e = np.vectorize(lambda c: CATEGORIES[c]["gift"])(category)
    hf = np.vectorize(lambda h: HOUR_FACTOR[h])(hour)
    bonus = np.vectorize(lambda c, h: CATEGORY_HOUR_BONUS.get(c, {}).get(h, 1.0))(category, hour)
    wf = np.asarray(WEEKDAY_FACTOR)[np.asarray(weekday)]

    disc_term = disc_e * (1 - np.exp(-np.asarray(card_disc) / 8))           # 한계효용 체감
    inst_term = inst_e * np.log1p(installment) / np.log(13)
    gift_term = gift_e * (1 - np.exp(-np.asarray(gift_pct) / 5))
    has_promo = (np.asarray(card_disc) > 0) | (np.asarray(installment) > 0) | (np.asarray(gift_pct) > 0)
    fatigue = 0.04 * np.asarray(recent_promo_cnt) * has_promo                 # 프로모션 피로도

    return (np.log(base_orders) + np.log(hf * bonus * wf)
            + disc_term + inst_term + gift_term - fatigue)


def true_return_rate(category, card_disc, gift_pct):
    base = np.vectorize(lambda c: CATEGORIES[c]["ret"])(np.asarray(category))
    return base * (1 + 0.8 * np.asarray(gift_pct) / 12 + 0.3 * np.asarray(card_disc) / 15)


def generate(start="2025-01-01", end="2026-09-30", slots_per_day=14, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    products = make_products(rng)
    dates = pd.date_range(start, end, freq="D")
    hours = list(HOUR_FACTOR.keys())

    rows = []
    last_promo_dates: dict[str, list] = {pid: [] for pid in products.product_id}
    for d in dates:
        for h in sorted(rng.choice(hours, size=slots_per_day, replace=False)):
            p = products.iloc[rng.integers(len(products))]
            prime = h in (20, 21, 22) or (d.weekday() >= 5 and 10 <= h <= 21)
            pricey = p.price >= 300_000
            # 교란: 황금 시간대·고가 상품에 프로모션이 더 자주 걸림
            p_disc = 0.25 + 0.25 * prime + 0.2 * pricey
            card_disc = int(rng.choice(CARD_DISC_LEVELS[1:])) if rng.random() < p_disc else 0
            card_share = float(rng.choice(CARD_SHARE_LEVELS)) if card_disc else 0.0
            p_inst = 0.15 + 0.5 * pricey
            installment = int(rng.choice(INSTALLMENT_LEVELS[1:])) if rng.random() < p_inst else 0
            p_gift = 0.3 + 0.25 * (p.category in ("뷰티", "식품")) + 0.1 * prime
            gift_pct = int(rng.choice(GIFT_LEVELS[1:])) if rng.random() < p_gift else 0

            recent = [x for x in last_promo_dates[p.product_id] if (d - x).days <= 14]
            last_promo_dates[p.product_id] = recent
            recent_cnt = len(recent)
            if card_disc or installment or gift_pct:
                last_promo_dates[p.product_id].append(d)

            mu = true_log_orders(p.base_orders, p.category, h, d.weekday(), card_disc, installment,
                                 gift_pct, recent_cnt)
            orders = int(rng.poisson(np.exp(float(mu) + rng.normal(0, 0.25))))
            rr = float(true_return_rate(p.category, card_disc, gift_pct))
            returns = int(rng.binomial(orders, min(rr, 0.95))) if orders else 0

            rows.append(dict(
                date=d.date().isoformat(), weekday=d.weekday(), hour=int(h),
                product_id=p.product_id, product_name=p.product_name, category=p.category,
                price=p.price, margin_rate=p.margin_rate,
                card_disc=card_disc, card_share=card_share, installment=installment, gift_pct=gift_pct,
                recent_promo_cnt=recent_cnt, orders=orders, returns=returns,
            ))

    df = pd.DataFrame(rows)
    df["gross_sales"] = df.orders * df.price
    return products, df


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out, exist_ok=True)
    products, df = generate()
    products.to_csv(f"{out}/products.csv", index=False)
    df.to_csv(f"{out}/broadcasts.csv", index=False)
    print(f"상품 {len(products)}개, 방송 {len(df):,}건 생성 → data/")
    print(df.head())
