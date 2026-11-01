"""
프로모션 반응 모델 학습.

- 주문 수 모델: 방송 조건(상품·시간대·요일·프로모션)으로 주문 수를 예측 (LightGBM)
  · 예상 범위(80% 구간): 학습에 쓰지 않은 '보정 기간'의 예측 오차 분포로 계산 (잔차 기반 구간)
  · 단조 제약(monotone constraint): "할인·무이자·사은품을 늘리면 주문이 줄지는 않는다"는 상식을 모델에 강제
- 반품률 모델: 프로모션 조건별 반품률 예측

학습 후 '정답을 알고 있는' 합성 데이터의 장점을 살려,
모델이 찾은 프로모션 효과(업리프트)가 진짜 효과와 얼마나 비슷한지 검증한다.
"""
from __future__ import annotations

import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from data_gen import true_log_orders, true_return_rate

ORDER_FEATURES = ["product_id", "category", "price", "hour", "weekday",
                  "card_disc", "installment", "gift_pct", "recent_promo_cnt"]
CAT_FEATURES = ["product_id", "category"]
# 단조 제약: +1 = 값이 커지면 예측도 커지거나 같음
MONOTONE = {"card_disc": 1, "installment": 1, "gift_pct": 1, "recent_promo_cnt": -1}
RETURN_FEATURES = ["product_id", "category", "card_disc", "installment", "gift_pct"]

CALIB_DATE = "2026-03-01"   # 이전: 학습
SPLIT_DATE = "2026-06-01"   # 3~5월: 예측 범위 보정, 6월 이후: 검증 (미래 데이터로 평가)


def _prep(df: pd.DataFrame, cat_levels: dict | None = None) -> pd.DataFrame:
    X = df.copy()
    for c in CAT_FEATURES:
        levels = cat_levels[c] if cat_levels else sorted(X[c].unique())
        X[c] = pd.Categorical(X[c], categories=levels)
    return X


def _lgb(features=ORDER_FEATURES, monotone=True):
    params = dict(objective="regression", n_estimators=600, learning_rate=0.03, num_leaves=31,
                  min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                  random_state=0, verbose=-1)
    if monotone:
        params["monotone_constraints"] = [MONOTONE.get(f, 0) for f in features]
    return lgb.LGBMRegressor(**params)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(data_path=f"{ROOT}/data/broadcasts.csv", products_path=f"{ROOT}/data/products.csv",
          out_dir=f"{ROOT}/models"):
    df = pd.read_csv(data_path)
    products = pd.read_csv(products_path)
    cat_levels = {c: sorted(df[c].unique()) for c in CAT_FEATURES}
    X = _prep(df, cat_levels)
    y = np.log1p(df.orders)
    tr = df.date < CALIB_DATE
    cal = (df.date >= CALIB_DATE) & (df.date < SPLIT_DATE)
    te = df.date >= SPLIT_DATE

    # 1) 주문 수 모델
    m_mid = _lgb().fit(X.loc[tr, ORDER_FEATURES], y[tr])
    # 보정 기간의 오차(로그 스케일) 분포 → 예상 범위와 평균 보정계수(smearing)
    resid = y[cal] - m_mid.predict(X.loc[cal, ORDER_FEATURES])
    q_lo, q_hi = float(np.quantile(resid, 0.1)), float(np.quantile(resid, 0.9))
    smear = float(np.mean(np.exp(resid)))

    # 2) 반품률 모델 (주문 수를 가중치로)
    rr = (df.returns / df.orders.clip(lower=1)).clip(0, 1)
    m_ret = _lgb(features=RETURN_FEATURES, monotone=False).fit(
        X.loc[tr, RETURN_FEATURES], rr[tr], sample_weight=df.orders[tr].clip(lower=1))

    # ── 검증 1: 미래 기간 예측 정확도 ──
    pred = np.expm1(m_mid.predict(X.loc[te, ORDER_FEATURES])) * smear
    actual = df.orders[te].to_numpy()
    wape = float(np.abs(pred - actual).sum() / actual.sum())
    log_mid = m_mid.predict(X.loc[te, ORDER_FEATURES])
    lo, hi = np.expm1(log_mid + q_lo), np.expm1(log_mid + q_hi)
    coverage = float(((actual >= lo) & (actual <= hi)).mean())

    # ── 검증 2: 프로모션 효과(업리프트)를 제대로 찾았나? (정답과 비교) ──
    test = df[te].merge(products[["product_id", "base_orders"]], on="product_id")
    uplift_rows = []
    levers = {"카드할인 10%": dict(card_disc=10), "무이자 12개월": dict(installment=12),
              "사은품 8%": dict(gift_pct=8)}
    for name, change in levers.items():
        # 피로도(recent_promo_cnt)는 0으로 고정 → 순수한 프로모션 효과만 비교
        base = test.assign(card_disc=0, installment=0, gift_pct=0, recent_promo_cnt=0)
        treat = base.assign(**change)
        for cat, g_base in base.groupby("category"):
            g_treat = treat.loc[g_base.index]
            args = lambda g: (g.base_orders, g.category, g.hour, g.weekday, g.card_disc,
                              g.installment, g.gift_pct, g.recent_promo_cnt)
            true_up = float(np.mean(np.exp(true_log_orders(*args(g_treat)) - true_log_orders(*args(g_base)))) - 1)
            pb = m_mid.predict(_prep(g_base, cat_levels)[ORDER_FEATURES])
            pt = m_mid.predict(_prep(g_treat, cat_levels)[ORDER_FEATURES])
            model_up = float(np.mean(np.expm1(pt)) / np.mean(np.expm1(pb)) - 1)
            uplift_rows.append(dict(lever=name, category=cat, true_uplift=round(true_up, 3),
                                    model_uplift=round(model_up, 3)))
    uplift = pd.DataFrame(uplift_rows)
    uplift_mae = float((uplift.true_uplift - uplift.model_uplift).abs().mean())

    # ── 참고: 단순 비교(평균 비교)의 함정 ──
    # 카드할인 한 방송 vs 안 한 방송의 평균 주문 수를 그냥 비교하면, 교란 요인(황금시간대) 때문에 효과가 부풀려진다
    naive = df[df.category == "가전"]
    naive_ratio = float(naive[naive.card_disc > 0].orders.mean() / naive[naive.card_disc == 0].orders.mean() - 1)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(dict(mid=m_mid, ret=m_ret, q_lo=q_lo, q_hi=q_hi, smear=smear, cat_levels=cat_levels),
                f"{out_dir}/models.joblib")
    importance = pd.Series(m_mid.booster_.feature_importance("gain"), index=ORDER_FEATURES)
    metrics = dict(
        train_rows=int(tr.sum()), calib_rows=int(cal.sum()), test_rows=int(te.sum()),
        calib_date=CALIB_DATE, split_date=SPLIT_DATE,
        test_wape=round(wape, 3), interval_coverage_80=round(coverage, 3),
        uplift_mae=round(uplift_mae, 3), naive_appliance_card_uplift=round(naive_ratio, 3),
        feature_importance=(importance / importance.sum()).round(3).sort_values(ascending=False).to_dict(),
    )
    with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    uplift.to_csv(f"{out_dir}/uplift_check.csv", index=False)
    return metrics, uplift


if __name__ == "__main__":
    metrics, uplift = train()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(uplift.to_string(index=False))
