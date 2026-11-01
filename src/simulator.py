"""
프로모션 시뮬레이터 — 조건별 주문·매출·이익을 계산하고 비교·최적화한다.

이익(공헌이익) 계산식 (가정은 ASSUMPTIONS에 명시):
  유지 주문      = 주문 × (1 - 반품률)
  순매출         = 유지 주문 × 판매가
  마진           = 순매출 × 마진율
  카드할인 부담  = 순매출 × 할인율 × (1 - 카드사 분담률)
  무이자 수수료  = 순매출 × 개월 수 × 월 수수료율
  사은품 원가    = 주문 × 판매가 × 사은품 비율 × (1 - 협력사 분담률)   (반품돼도 회수 못한다고 가정)
  반품 처리비    = 주문 × 반품률 × 건당 처리비
  공헌이익       = 마진 - 카드할인 부담 - 무이자 수수료 - 사은품 원가 - 반품 처리비
"""
from __future__ import annotations

import itertools
import os

import joblib
import numpy as np
import pandas as pd

from data_gen import (CARD_DISC_LEVELS, GIFT_LEVELS, INSTALLMENT_FEE_PER_MONTH, INSTALLMENT_LEVELS,
                      RETURN_HANDLING_COST)
from model import ORDER_FEATURES, RETURN_FEATURES, _prep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
ASSUMPTIONS = {
    "무이자 할부 수수료(개월당, 판매가 대비)": f"{INSTALLMENT_FEE_PER_MONTH:.2%}",
    "반품 1건 처리비": f"{RETURN_HANDLING_COST:,}원",
    "사은품": "반품 시 회수하지 못한다고 가정, 원가 중 협력사 부담분은 제외",
    "판매가·마진율": "마진율 = 판매가 대비 홈쇼핑 몫(판매수수료 등) 가정값",
    "데이터": "합성(가상) 데이터 — 본선에서 실제 데이터로 교체",
}


def parse_weekday(w) -> int:
    if isinstance(w, (int, np.integer)):
        return int(w) % 7
    w = str(w).strip()
    if w.isdigit():
        return int(w) % 7
    return WEEKDAYS.index(w[0])


class Simulator:
    def __init__(self, root: str = ROOT):
        self.products = pd.read_csv(f"{root}/data/products.csv")
        self.history = pd.read_csv(f"{root}/data/broadcasts.csv")
        self.m = joblib.load(f"{root}/models/models.joblib")

    # ── 조회 ──────────────────────────────────────────────
    def list_products(self, category: str | None = None, keyword: str | None = None) -> pd.DataFrame:
        p = self.products
        if category:
            p = p[p.category == category]
        if keyword:
            p = p[p.product_name.str.contains(keyword) | p.category.str.contains(keyword)]
        return p[["product_id", "product_name", "category", "price", "margin_rate"]]

    def product(self, product_id: str) -> pd.Series:
        hit = self.products[self.products.product_id == product_id]
        if hit.empty:
            raise ValueError(f"상품 ID를 찾을 수 없습니다: {product_id}")
        return hit.iloc[0]

    def product_history(self, product_id: str) -> dict:
        h = self.history[self.history.product_id == product_id].copy()
        p = self.product(product_id)
        h["promo"] = np.select(
            [h.card_disc > 0, h.installment > 0, h.gift_pct > 0], ["카드할인", "무이자", "사은품"], "없음")
        by_promo = h.groupby("promo").agg(방송수=("orders", "size"), 평균주문=("orders", "mean")).round(0)
        best = h.sort_values("orders", ascending=False).head(3)[
            ["date", "hour", "card_disc", "installment", "gift_pct", "orders"]]
        return dict(product=f"{p.product_name}({p.category}, {p.price:,.0f}원)", broadcasts=len(h),
                    avg_orders=round(h.orders.mean(), 0),
                    by_main_promo=by_promo.reset_index().to_dict("records"),
                    top3=best.to_dict("records"))

    # ── 시뮬레이션 ─────────────────────────────────────────
    def _frame(self, product_id, weekday, hour, scenarios: list[dict]) -> pd.DataFrame:
        p = self.product(product_id)
        rows = []
        for s in scenarios:
            rows.append(dict(
                scenario=s.get("name") or self.describe(s),
                product_id=p.product_id, category=p.category, price=p.price, margin_rate=p.margin_rate,
                hour=int(hour), weekday=parse_weekday(weekday),
                card_disc=float(s.get("card_disc", 0)), card_share=float(s.get("card_share", 0.5)),
                gift_vendor_share=float(s.get("gift_vendor_share", 0.0)),
                installment=float(s.get("installment", 0)), gift_pct=float(s.get("gift_pct", 0)),
                recent_promo_cnt=int(s.get("recent_promo_cnt", 0)),
            ))
        return pd.DataFrame(rows)

    @staticmethod
    def describe(s: dict) -> str:
        parts = []
        if s.get("card_disc"):
            parts.append(f"카드 {s['card_disc']:g}%")
        if s.get("installment"):
            parts.append(f"무이자 {s['installment']:g}개월")
        if s.get("gift_pct"):
            parts.append(f"사은품 {s['gift_pct']:g}%")
        return " + ".join(parts) or "프로모션 없음"

    def _evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        m = self.m
        X = _prep(df, m["cat_levels"])
        log_mid = m["mid"].predict(X[ORDER_FEATURES])
        orders = np.expm1(log_mid) * m["smear"]
        lo, hi = np.expm1(log_mid + m["q_lo"]), np.expm1(log_mid + m["q_hi"])
        rr = np.clip(m["ret"].predict(X[RETURN_FEATURES]), 0, 0.9)

        out = df[["scenario", "card_disc", "card_share", "gift_vendor_share", "installment", "gift_pct"]].copy()
        price, margin = df.price.to_numpy(), df.margin_rate.to_numpy()

        def money(o):
            kept = o * (1 - rr)
            net_sales = kept * price
            disc = net_sales * df.card_disc / 100 * (1 - df.card_share)
            inst = net_sales * df.installment * INSTALLMENT_FEE_PER_MONTH
            gift = o * price * df.gift_pct / 100 * (1 - df.gift_vendor_share)
            ret_cost = o * rr * RETURN_HANDLING_COST
            profit = net_sales * margin - disc - inst - gift - ret_cost
            return net_sales, disc, inst, gift, ret_cost, profit

        net, disc, inst, gift, ret_cost, profit = money(orders)
        out["예상주문"] = orders.round(0)
        out["주문_하한"], out["주문_상한"] = lo.round(0), hi.round(0)
        out["반품률"] = rr.round(3)
        out["순매출"] = net.round(-3)
        out["카드할인부담"], out["무이자수수료"], out["사은품원가"], out["반품처리비"] = (
            disc.round(-3), inst.round(-3), gift.round(-3), ret_cost.round(-3))
        out["프로모션비용"] = (disc + inst + gift).round(-3)
        out["공헌이익"] = profit.round(-3)
        out["이익_하한"] = money(lo)[-1].round(-3)
        out["이익_상한"] = money(hi)[-1].round(-3)
        return out

    def simulate(self, product_id, weekday, hour, scenarios: list[dict], add_baseline=True) -> pd.DataFrame:
        """여러 프로모션 시나리오를 같은 방송 조건에서 비교. '프로모션 없음' 기준선 대비 증감도 계산."""
        sc = list(scenarios)
        if add_baseline and not any(self.describe(s) == "프로모션 없음" for s in sc):
            sc = [{"name": "기준(프로모션 없음)"}] + sc
        res = self._evaluate(self._frame(product_id, weekday, hour, sc))
        base = res.iloc[0]
        res["매출증감률"] = (res.순매출 / base.순매출 - 1).round(3)
        res["이익증감"] = (res.공헌이익 - base.공헌이익).round(-3)
        # 프로모션비 1원당 추가 이익 (ROI)
        res["프로모션ROI"] = np.where(res.프로모션비용 > 0, (res.이익증감 / res.프로모션비용).round(2), np.nan)
        return res

    def optimize(self, product_id, weekday, hour, objective="profit", card_share=0.5, gift_vendor_share=0.0,
                 max_promo_cost_rate: float | None = None, top_n=5, recent_promo_cnt=0) -> pd.DataFrame:
        """모든 프로모션 조합(5×5×5=125개)을 시뮬레이션해 목표 기준 상위 조합을 찾는다.
        objective: 'profit'(공헌이익 최대) | 'sales'(순매출 최대)
        max_promo_cost_rate: 프로모션 비용이 순매출의 몇 % 이하여야 하는지 (예: 0.08)"""
        combos = [dict(card_disc=c, installment=i, gift_pct=g, card_share=card_share, gift_vendor_share=gift_vendor_share,
                       recent_promo_cnt=recent_promo_cnt)
                  for c, i, g in itertools.product(CARD_DISC_LEVELS, INSTALLMENT_LEVELS, GIFT_LEVELS)]
        res = self.simulate(product_id, weekday, hour, combos, add_baseline=False)
        base = res[(res.card_disc == 0) & (res.installment == 0) & (res.gift_pct == 0)].iloc[0]
        if max_promo_cost_rate is not None:
            res = res[res.프로모션비용 <= res.순매출 * max_promo_cost_rate]
        key = "공헌이익" if objective == "profit" else "순매출"
        res = res.sort_values(key, ascending=False).head(top_n).copy()
        res["매출증감률"] = (res.순매출 / base.순매출 - 1).round(3)
        res["이익증감"] = (res.공헌이익 - base.공헌이익).round(-3)
        return res


if __name__ == "__main__":
    sim = Simulator()
    print(sim.list_products("가전").head())
    r = sim.simulate("P001", "화", 21, [dict(card_disc=10), dict(installment=12), dict(gift_pct=8)])
    pd.set_option("display.width", 250)
    print(r[["scenario", "예상주문", "주문_하한", "주문_상한", "순매출", "프로모션비용", "공헌이익", "매출증감률", "이익증감", "프로모션ROI"]])
    print(sim.optimize("P001", "화", 21)[["scenario", "순매출", "공헌이익", "매출증감률", "이익증감"]])
    print(sim.optimize("P009", "토", 10)[["scenario", "순매출", "공헌이익", "매출증감률", "이익증감"]])
