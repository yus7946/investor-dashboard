"""機関投資家の資金フロー（買い集め / 利確・売り抜け）を推測するスコア。

日足OHLCVから、大口の需給を反映しやすい以下の指標を合成する。すべて追加通信ゼロ。
  - CMF (Chaikin Money Flow): 高値圏で買われているか安値圏で売られているか。
      正=安値でも買われる（買い集めの兆候）／負=高値でも売られる（売り抜けの兆候）。
  - OBV傾き: 出来高の方向性（上昇日に出来高が集中するか）。
  - MFI (Money Flow Index): 出来高加重RSI。資金の流入出の過熱度。
  - 移動VWAP乖離: 直近20日の出来高加重平均価格に対し、現値が上か下か
      （機関の平均取得コストを上回って推移＝買い持ちが優勢の目安）。
  - 出来高方向: 直近5日の上昇日出来高 vs 下落日出来高の偏り。

これらは「推測」であり確定情報ではない。断定を避け、"兆候" として提示する前提の指標。
"""


def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if f == f else default  # NaN除外
    except Exception:
        return default


def compute_smart_money(hist) -> dict | None:
    """hist: yfinanceのhistory(6mo)相当のDataFrame（High/Low/Close/Volume）。"""
    try:
        if hist is None or len(hist) < 30:
            return None
        high, low, close, vol = hist["High"], hist["Low"], hist["Close"], hist["Volume"]

        # --- CMF(20) ---
        rng = (high - low).replace(0, 1e-9)
        mfm = ((close - low) - (high - close)) / rng
        cmf = _safe((mfm * vol).rolling(20).sum().iloc[-1] / vol.rolling(20).sum().iloc[-1])

        # --- OBV傾き（直近20日の平均日次変化を出来高規模で正規化） ---
        step = (close.diff() > 0).astype(int) - (close.diff() < 0).astype(int)
        obv = (vol * step).cumsum()
        obv_slope = _safe(obv.diff().tail(20).mean())
        avg_vol = _safe(vol.tail(20).mean(), 1.0) or 1.0
        obv_norm = max(-1.0, min(1.0, obv_slope / avg_vol))  # -1〜1

        # --- MFI(14) ---
        tp = (high + low + close) / 3
        mf = tp * vol
        pos = mf.where(tp.diff() > 0, 0).rolling(14).sum()
        neg = mf.where(tp.diff() < 0, 0).rolling(14).sum().replace(0, 1e-9)
        mfi = _safe((100 - 100 / (1 + pos / neg)).iloc[-1], 50.0)

        # --- 移動VWAP(20)乖離 ---
        vwap20 = _safe((tp * vol).rolling(20).sum().iloc[-1] / vol.rolling(20).sum().iloc[-1])
        price = _safe(close.iloc[-1])
        vwap_dev = (price - vwap20) / vwap20 if vwap20 else 0.0

        # --- 出来高方向（直近5日：上昇日出来高 - 下落日出来高 の偏り） ---
        recent = hist.tail(5)
        up_vol = _safe(recent["Volume"][recent["Close"].diff() > 0].sum())
        dn_vol = _safe(recent["Volume"][recent["Close"].diff() < 0].sum())
        vol_dir = (up_vol - dn_vol) / (up_vol + dn_vol) if (up_vol + dn_vol) else 0.0

        # --- 各指標を -1〜+1 に正規化して合成 ---
        s_cmf = max(-1.0, min(1.0, cmf / 0.25))          # CMF ±0.25 で飽和
        s_obv = obv_norm
        s_mfi = max(-1.0, min(1.0, (mfi - 50) / 30))     # MFI 20-80 を -1〜1
        s_vwap = max(-1.0, min(1.0, vwap_dev / 0.05))    # ±5%乖離で飽和
        s_voldir = max(-1.0, min(1.0, vol_dir))

        # CMFとOBVを主軸に（大口需給を最も反映）、他を補助
        score = (s_cmf * 0.30 + s_obv * 0.25 + s_mfi * 0.15 + s_vwap * 0.15 + s_voldir * 0.15)
        score = round(max(-1.0, min(1.0, score)), 3)

        # --- 判定ラベルと根拠 ---
        if score >= 0.25:
            label, tone = "買い集めの兆候", "pos"
        elif score <= -0.25:
            label, tone = "利確・売り抜けの兆候", "neg"
        else:
            label, tone = "中立", "neu"

        factors = []
        if s_cmf >= 0.3:
            factors.append("安値圏でも買われている（CMFプラス）")
        elif s_cmf <= -0.3:
            factors.append("高値圏で売られている（CMFマイナス）")
        if s_obv >= 0.2:
            factors.append("上昇日に出来高が集中（OBV上向き）")
        elif s_obv <= -0.2:
            factors.append("下落日に出来高が集中（OBV下向き）")
        if s_vwap >= 0.3:
            factors.append("直近平均コスト(VWAP)を上回って推移")
        elif s_vwap <= -0.3:
            factors.append("直近平均コスト(VWAP)を下回って推移")
        if s_voldir >= 0.4:
            factors.append("直近5日は買い方向の出来高が優勢")
        elif s_voldir <= -0.4:
            factors.append("直近5日は売り方向の出来高が優勢")

        return {
            "score": score,          # -1（売り抜け）〜 +1（買い集め）
            "label": label,
            "tone": tone,
            "cmf": round(cmf, 3),
            "mfi": round(mfi, 1),
            "obvSlope": round(obv_norm, 2),
            "vwapDevPct": round(vwap_dev * 100, 1),
            "factors": factors,
        }
    except Exception:
        return None
