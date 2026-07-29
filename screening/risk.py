"""リスク管理レイヤー：1トレードの負けを小さく・浅く固定するための「安全度」と運用ルールを算出する。

設計思想（正直な前提）:
  「勝てる」ことは保証できない。プロが資産を守る要は「1回の負けを資金の1〜2%に固定し、
  退場しないこと」にある。ここでは各銘柄の負けやすさ・傷の深さの要因を、すでに取得済みの
  実データ（変動率・機関の売り抜け兆候・過熱・空売り・決算接近）だけで定量化する。追加通信ゼロ。

  安全度(0〜100, 高いほど低リスク)は将来の勝敗予測ではなく、
  「今エントリーした場合に想定される値動きの荒さ・イベントリスクの大きさ」の相対的な目安。
  すべて要因を明示し、ブラックボックスにしない。
"""

# 1トレードで取るリスクの上限（資金に対する割合）。プロの標準的な規律。
RISK_BUDGET_CONSERVATIVE = 0.01  # 1%（守り重視）
RISK_BUDGET_STANDARD = 0.02      # 2%（標準）


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def compute_risk(stock: dict) -> dict:
    """stockにひもづく実データから安全度と要因を返す。stock['risk']へ格納する想定。"""
    vol = stock.get("volatility")            # 年率換算ボラティリティ（0.30=30%）
    rsi = stock.get("rsi") or 50
    sm = (stock.get("smart_money") or {}).get("score")   # -1(売り抜け)〜+1(買い集め)
    sr = stock.get("short_ratio")            # 空売り残高%（開示銘柄のみ）
    earn = stock.get("earnings") or {}
    days_to = earn.get("daysTo")             # 決算までの営業日数（負=発表済み直後）
    fc = stock.get("forecast") or {}
    plan = fc.get("plan") or {}
    entry, stop = plan.get("entry"), plan.get("stop")

    safety = 100.0
    penalties = []   # マイナス要因（リスク）
    supports = []    # プラス要因（安心材料）

    # ① 値動きの荒さ（年率ボラ）。20%を基準に、大きいほど減点（最大-40）。
    if vol is not None:
        volp = _clip((vol - 0.20) * 100, 0, 40)
        safety -= volp
        if vol >= 0.45:
            penalties.append(f"値動きが大きい（年率{vol*100:.0f}%）→ 建玉は小さめに")
        elif vol <= 0.22:
            supports.append(f"値動きが比較的穏やか（年率{vol*100:.0f}%）")

    # ② 機関の売り抜け兆候（スマートマネー負）。降りている所を買わない（最大-25）。
    if sm is not None:
        if sm < 0:
            safety -= _clip(-sm * 25, 0, 25)
            if sm <= -0.25:
                penalties.append("機関の売り抜け兆候（新規は見送り推奨）")
        elif sm >= 0.25:
            supports.append("機関の買い集め兆候")

    # ③ 過熱（高RSI）。高値づかみで浅い押しでも損切りに当たりやすい（最大-20）。
    if rsi >= 68:
        safety -= _clip((rsi - 68) * 1.3, 0, 20)
        if rsi >= 75:
            penalties.append(f"買われすぎ（RSI{rsi:.0f}）→ 反落で損切りに当たりやすい")
    elif rsi <= 35:
        supports.append(f"売られすぎ圏（RSI{rsi:.0f}・反発余地）")

    # ④ 空売り残高。踏み上げ・急落どちらの急変要因にもなる（最大-12）。
    if sr is not None and sr >= 2.0:
        safety -= _clip((sr - 2.0) * 3, 0, 12)
        penalties.append(f"空売り残高{sr}%（急変動リスク）")

    # ⑤ 決算接近。発表前後は窓（ギャップ）で損切り幅を超えて飛ぶことがある（最大-25）。
    if isinstance(days_to, (int, float)):
        if -1 <= days_to <= 3:
            safety -= 25
            penalties.append(f"決算が至近（{'本日' if days_to == 0 else f'あと{int(days_to)}営業日'}）→ 窓リスク大・またぎ回避推奨")
        elif 3 < days_to <= 7:
            safety -= 12
            penalties.append(f"決算接近（あと{int(days_to)}営業日）→ 建玉は控えめに")

    safety = round(_clip(safety, 0, 100))
    level = "低" if safety >= 70 else ("中" if safety >= 45 else "高")

    # 損切り幅（現値ではなくエントリーからの下落率）。ポジションサイズ計算の分母。
    stop_pct = None
    if entry and stop and entry > stop:
        stop_pct = round((entry - stop) / entry * 100, 2)

    return {
        "safety": safety,          # 0〜100（高いほど低リスク）
        "level": level,            # 低/中/高（リスクの高さ）
        "penalties": penalties,
        "supports": supports,
        "stopPct": stop_pct,       # エントリーから損切りまでの下落率(%)
        "rr": plan.get("rr"),      # 利益:損失の比率
    }


def annotate_risk(stocks: list[dict]) -> list[dict]:
    for s in stocks:
        try:
            s["risk"] = compute_risk(s)
        except Exception:
            s["risk"] = None
    return stocks


def build_low_risk_plan(stocks: list[dict], regime: str = "unknown",
                        monthly_target: int = 200000) -> dict:
    """負けを固定する運用ルールと、条件を満たす『低リスク買い候補バスケット』を返す。

    選定条件（すべて満たすもののみ・厳しめ）:
      - シグナルが「買い」または「強い買い」
      - 安全度 60 以上
      - 機関の売り抜け兆候がない（スマートマネー >= 0）
      - 決算が至近（3営業日以内）でない
      - 利益:損失の比率(RR)が 1.5 以上（勝率が5割でも負けにくい設計）
    """
    basket = []
    for s in stocks:
        risk = s.get("risk") or {}
        sm = (s.get("smart_money") or {}).get("score")
        earn = s.get("earnings") or {}
        days_to = earn.get("daysTo")
        plan = (s.get("forecast") or {}).get("plan") or {}
        rr = plan.get("rr")
        if not plan.get("entry") or not plan.get("stop"):
            continue  # 具体的な損切りラインを提示できない銘柄は候補にしない
        if s.get("signal") not in ("買い", "強い買い"):
            continue
        if (risk.get("safety") or 0) < 60:
            continue
        if sm is not None and sm < 0:
            continue
        if isinstance(days_to, (int, float)) and -1 <= days_to <= 3:
            continue
        if rr is not None and rr < 1.5:
            continue
        basket.append({
            "ticker": s["ticker"],
            "name": s["name"],
            "signal": s["signal"],
            "safety": risk.get("safety"),
            "stopPct": risk.get("stopPct"),
            "rr": rr,
            "entry": plan.get("entry"),
            "stop": plan.get("stop"),
            "target": plan.get("target"),
        })
    # 安全度優先で並べ、最大5銘柄（分散しつつ絞る）
    basket.sort(key=lambda x: (x["safety"] or 0), reverse=True)
    basket = basket[:5]

    # 地合いに応じた新規建ての姿勢
    if regime == "bear":
        stance = "地合いが下落基調のため、新規は数を絞り現金比率を高めに。無理な追いは避ける。"
    elif regime == "bull":
        stance = "地合いは上昇基調。ただし1トレードのリスク上限は変えず、規律を優先。"
    else:
        stance = "地合いは中立。1トレードのリスク上限を守り、条件を満たす銘柄だけを対象に。"

    rules = [
        "1トレードで取るリスクは資金の1〜2%まで（損切り幅から逆算した株数に抑える）。",
        "エントリーと同時に損切り価格を必ず置く（下の各銘柄の損切りラインを使用）。",
        "利益:損失の比率が1.5:1以上の場面だけ取る（勝率5割でも負けにくくするため）。",
        "機関の売り抜け兆候・決算至近・買われすぎ（RSI高）の銘柄は新規で買わない。",
        "同時に持つ銘柄を分散し、1銘柄への集中を避ける。",
    ]

    return {
        "basket": basket,
        "rules": rules,
        "stance": stance,
        "riskBudget": {"conservative": RISK_BUDGET_CONSERVATIVE, "standard": RISK_BUDGET_STANDARD},
        "monthlyTarget": monthly_target,
        "note": (
            "これは『負けを小さく固定するための規律』であり、利益を保証するものではありません。"
            "月々の目標額は相場次第で達成できない月もあります。安全度は将来の勝敗予測ではなく、"
            "今の値動きの荒さ・イベントリスクの相対的な目安です。"
        ),
    }
