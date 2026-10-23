"""
BTC LONG-ONLY: PERBANDINGAN METODE PENGURANG MDD (bukan mencari alpha).

Semua metode menghasilkan bobot BTC w in [0,1] (sisanya kas, return kas = CASH_ANN, default 0).
Sinyal dihitung di close hari t, posisi dipegang dari open t+1 (open-to-open), return COMPOUNDING.
Biaya: FEE per sisi x turnover (default 0.10%); kolom CAGR_f20 = biaya 0.20%.

KELUARGA METODE (jumlah konfigurasi dicetak di output):
  trend   : SMA, EMA, SMA cross, Donchian long-only, TSMOM, SMA dengan hysteresis band, SMA cek mingguan
  stop    : trailing stop dari puncak (+ aturan masuk lagi)
  scaling : exposure turun mengikuti drawdown dari ATH, volatility targeting
  gabungan: SMA200 x vol-target, ensemble SMA (50..300), ensemble TSMOM (30..252)
  kontrol : alokasi statis 25/50/75% (garis dasar "tinggal kurangi bobot")

METRIK KUNCI
  MDD, CAGR, Calmar, Sharpe, turnover/tahun, avg exposure, up/down capture,
  ret_BHcrash = return strategi dari puncak ke dasar drawdown terbesar BTC buy&hold (periode bear).
  Calmar_static = Calmar alokasi statis dengan bobot = rata-rata exposure strategi. Kalau strategi tidak
  mengalahkannya, timing-nya tidak menambah apa pun dibanding sekadar memegang BTC lebih sedikit.
  USEFUL   : |MDD| <= 70% dari |MDD| buy&hold DAN Calmar > Calmar buy&hold.
  BEATS_STATIC : Calmar > Calmar_static.

ATURAN: JANGAN pilih satu konfigurasi terbaik. Lihat ringkasan per keluarga (plateau): apakah hasil bertahan
  di banyak parameter. Sampel hanya mencakup 1 bear market besar (2022) kecuali kamu menambah data spot
  (BTC_SPOT_DIR) yang memuat 2018. Itu bukti sangat tipis. OOS 2024+ tidak disentuh (END 2023-12-31).
"""
import os, math
import numpy as np
import pandas as pd
import explore_behavior as eb

FEE = float(os.environ.get("BTC_FEE", 0.0010))
CASH_ANN = float(os.environ.get("CASH_ANN", 0.0))
WARMUP = 300
RESULT_DIR = os.path.join(eb.BASE_DIR, "Results", "btc_mdd_reduction")
USEFUL_MDD_RATIO = 0.70


# ---------- data ----------
def load_btc():
    d = os.environ.get("BTC_SPOT_DIR") or eb.PERP_DIR
    files = eb.find_files(d).get("BTCUSDT")
    if not files:
        raise SystemExit(f"BTCUSDT tidak ditemukan di {d}")
    parts = []
    for f in files:
        x = pd.read_csv(f)
        parts.append(pd.DataFrame({"t": eb.to_time(x[eb.pick(x.columns, eb.TIME_COLS)]),
                                   "open": x[eb.pick(x.columns, ["open"])], "close": x[eb.pick(x.columns, ["close"])]}))
    k = pd.concat(parts).drop_duplicates("t").set_index("t").sort_index()
    k = k[k.index < eb.END + pd.Timedelta(days=1)]
    if k.index.to_series().diff().median() >= pd.Timedelta(hours=20):
        dd = k.copy()
        dd.index = dd.index.normalize()
    else:
        dd = k.resample("D").agg({"open": "first", "close": "last"})
        dd = dd[k["open"].resample("D").count() >= 20]
    dd = dd.dropna()
    print(f"BTC: {dd.index[0].date()} s/d {dd.index[-1].date()} ({len(dd)} hari) dari {d}")
    return dd["open"], dd["close"]


# ---------- sinyal ----------
def sma_f(C, L):
    return (C > C.rolling(L).mean()).astype(float)


def ema_f(C, L):
    return (C > C.ewm(span=L, adjust=False).mean()).astype(float)


def cross_f(C, a, b):
    return (C.rolling(a).mean() > C.rolling(b).mean()).astype(float)


def tsmom_f(C, L):
    return ((C / C.shift(L) - 1) > 0).astype(float)


def donchian_f(C, N):
    hh, ll = C.shift(1).rolling(N).max().values, C.shift(1).rolling(max(N // 2, 2)).min().values
    c, s, out = C.values, 0, np.zeros(len(C))
    for t in range(len(c)):
        if s == 0 and c[t] > hh[t]:
            s = 1
        elif s == 1 and c[t] < ll[t]:
            s = 0
        out[t] = s
    return pd.Series(out, index=C.index)


def ddstop_f(C, x, N):
    re = C.shift(1).rolling(N).max().values
    c, s, peak, out = C.values, 1, C.values[0], np.zeros(len(C))
    for t in range(len(c)):
        if s == 1:
            peak = max(peak, c[t])
            if c[t] < peak * (1 - x):
                s = 0
        elif c[t] > re[t]:
            s, peak = 1, c[t]
        out[t] = s
    return pd.Series(out, index=C.index)


def hyst_f(C, L, b):
    m = C.rolling(L).mean().values
    c, s, out = C.values, 0, np.zeros(len(C))
    for t in range(len(c)):
        if not np.isnan(m[t]):
            if s == 0 and c[t] > m[t] * (1 + b):
                s = 1
            elif s == 1 and c[t] < m[t] * (1 - b):
                s = 0
        out[t] = s
    return pd.Series(out, index=C.index)


def weekly_f(C, L):
    sig = sma_f(C, L)
    return sig.where(C.index.dayofweek == 6).ffill().fillna(0.0)


def ddscale_f(C, ddmax):
    dd = C / C.cummax() - 1
    return (1 + dd / ddmax).clip(0, 1)


def volt_f(C, target):
    sig = C.pct_change().rolling(30, min_periods=20).std() * math.sqrt(365)
    return (target / sig).clip(0, 1).fillna(0.0)


def make_families(C):
    F = {}
    F["SMA"] = {f"SMA{L}": sma_f(C, L) for L in (50, 100, 150, 200, 250, 300)}
    F["EMA"] = {f"EMA{L}": ema_f(C, L) for L in (50, 100, 150, 200, 250, 300)}
    F["SMA cross"] = {f"cross{a}/{b}": cross_f(C, a, b) for a, b in ((20, 50), (20, 100), (50, 100), (50, 150), (50, 200), (100, 200))}
    F["Donchian long-only"] = {f"don{N}/{N//2}": donchian_f(C, N) for N in (50, 100, 150, 200)}
    F["TSMOM"] = {f"tsmom{L}": tsmom_f(C, L) for L in (30, 60, 90, 120, 180, 252)}
    F["SMA hysteresis"] = {f"SMA{L}±{int(b*100)}%": hyst_f(C, L, b) for L in (150, 200) for b in (0.03, 0.05, 0.10)}
    F["SMA weekly check"] = {f"SMAw{L}": weekly_f(C, L) for L in (100, 150, 200, 250)}
    F["Trailing stop"] = {f"stop{int(x*100)}%/re{N}": ddstop_f(C, x, N) for x in (0.15, 0.20, 0.25, 0.30, 0.40) for N in (60, 120)}
    F["DD-scaled exposure"] = {f"ddscale{int(m*100)}%": ddscale_f(C, m) for m in (0.30, 0.40, 0.50, 0.60)}
    F["Vol target"] = {f"volT{int(t*100)}%": volt_f(C, t) for t in (0.30, 0.40, 0.50, 0.60, 0.80)}
    s200 = sma_f(C, 200)
    F["SMA200 x volT"] = {f"SMA200xvolT{int(t*100)}%": s200 * volt_f(C, t) for t in (0.40, 0.60, 0.80)}
    F["Ensemble"] = {"ens_SMA(50-300)": pd.concat([sma_f(C, L) for L in (50, 100, 150, 200, 250, 300)], axis=1).mean(axis=1),
                     "ens_TSMOM(30-252)": pd.concat([tsmom_f(C, L) for L in (30, 60, 90, 120, 180, 252)], axis=1).mean(axis=1)}
    F["Kontrol statis"] = {f"static{int(w*100)}%": pd.Series(w, index=C.index) for w in (0.25, 0.50, 0.75)}
    return F


# ---------- evaluasi ----------
def cagr_mdd(r):
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (365 / n) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return cagr, mdd


def evaluate(w, R, start, fee, cash_d):
    Ws = w.shift(1).fillna(0.0)[start:]
    Rs = R[start:]
    dW = Ws.diff().abs()
    dW.iloc[0] = Ws.iloc[0]
    ret = Ws * Rs + (1 - Ws) * cash_d - dW * fee
    return ret, Ws, dW


def main_analysis(O, C, outdir=None, quiet=False):
    idx = C.index
    start = idx[0] + pd.Timedelta(days=WARMUP)
    R = (O.shift(-1) / O - 1)
    R = R.iloc[:-1]
    C, O = C.loc[R.index], O.loc[R.index]
    cash_d = (1 + CASH_ANN) ** (1 / 365) - 1
    fams = make_families(C)
    nconf = sum(len(v) for v in fams.values())

    bh_ret, bh_W, _ = evaluate(pd.Series(1.0, index=C.index), R, start, FEE, cash_d)
    bh_cagr, bh_mdd = cagr_mdd(bh_ret)
    bh_cal = bh_cagr / abs(bh_mdd)
    eq_bh = (1 + bh_ret).cumprod()
    peak = (eq_bh.cummax())
    trough_d = (eq_bh / peak - 1).idxmin()
    peak_d = eq_bh[:trough_d].idxmax()
    up, dn = bh_ret > 0, bh_ret < 0
    print(f"\nEvaluasi: {bh_ret.index[0].date()} s/d {bh_ret.index[-1].date()} ({len(bh_ret)} hari). Konfigurasi diuji: {nconf}.")
    print(f"BUY&HOLD BTC: CAGR={bh_cagr*100:.1f}% MDD={bh_mdd*100:.1f}% Calmar={bh_cal:.2f} | "
          f"drawdown terbesar: puncak {peak_d.date()} -> dasar {trough_d.date()}")

    rows, rets = [], {}
    for fam, d in fams.items():
        for name, w in d.items():
            ret, Ws, dW = evaluate(w, R, start, FEE, cash_d)
            ret2, _, _ = evaluate(w, R, start, 0.0020, cash_d)
            cagr, mdd = cagr_mdd(ret)
            sd = ret.std()
            eq = (1 + ret).cumprod()
            avg = Ws.mean()
            sret = avg * R[start:] + (1 - avg) * cash_d
            sc, sm = cagr_mdd(sret)
            cal = cagr / abs(mdd) if mdd < 0 else np.nan
            cal_s = sc / abs(sm) if sm < 0 else np.nan
            crash = eq[trough_d] / eq[peak_d] - 1
            years = len(ret) / 365
            rows.append(dict(family=fam, strategy=name, CAGR=cagr * 100, MDD=mdd * 100, Calmar=cal,
                             Sharpe=ret.mean() / sd * math.sqrt(365) if sd > 0 else np.nan,
                             ret_BHcrash=crash * 100, exposure=avg * 100, turn_yr=dW.sum() / years,
                             up_cap=ret[up].mean() / bh_ret[up].mean() * 100, down_cap=ret[dn].mean() / bh_ret[dn].mean() * 100,
                             CAGR_f20=cagr_mdd(ret2)[0] * 100, Calmar_static=cal_s,
                             USEFUL=bool(abs(mdd) <= USEFUL_MDD_RATIO * abs(bh_mdd) and cal > bh_cal),
                             BEATS_STATIC=bool(cal > cal_s)))
            rets[name] = ret
    res = pd.DataFrame(rows)
    if quiet:
        return res, rets, (bh_cagr, bh_mdd)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== SEMUA KONFIGURASI (CAGR, MDD, ret_BHcrash, exposure, capture dalam %; turn_yr = pergantian posisi penuh/tahun) ===")
    print(res.drop(columns="family").round(2).to_string(index=False))

    print("\n=== RINGKASAN PER KELUARGA (plateau: lihat % USEFUL, bukan satu angka terbaik) ===")
    g = res.groupby("family", sort=False)
    summ = pd.DataFrame({"n": g.size(), "MDD_med": g.MDD.median(), "MDD_worst": g.MDD.min(), "CAGR_med": g.CAGR.median(),
                         "Calmar_med": g.Calmar.median(), "crash_med": g.ret_BHcrash.median(),
                         "pct_USEFUL": g.USEFUL.mean() * 100, "pct_BEATS_STATIC": g.BEATS_STATIC.mean() * 100})
    print(summ.round(2).to_string())
    print(f"\nBuy&hold: MDD={bh_mdd*100:.1f}% CAGR={bh_cagr*100:.1f}% Calmar={bh_cal:.2f}. "
          f"USEFUL = |MDD|<={int(USEFUL_MDD_RATIO*100)}% MDD BH & Calmar>BH.")
    print("Keluarga yang layak diperhatikan: pct_USEFUL tinggi (>=60%) DAN pct_BEATS_STATIC tinggi.")
    print("Ingat: sampel = 1 bear market. Semua hasil adalah gambaran, bukan bukti.")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        res.to_csv(os.path.join(outdir, "btc_mdd_all.csv"), index=False)
        summ.to_csv(os.path.join(outdir, "btc_mdd_family_summary.csv"))
        pd.DataFrame(rets).assign(BUYHOLD=bh_ret).to_csv(os.path.join(outdir, "daily_returns.csv"))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            show = ["SMA200", "ens_SMA(50-300)", "volT50%", "ddscale50%", "stop25%/re60", "static50%"]
            fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            for nm, r in [("BUY&HOLD", bh_ret)] + [(s, rets[s]) for s in show if s in rets]:
                e = (1 + r).cumprod()
                ax[0].plot(e.index, e, label=nm, lw=2.2 if nm == "BUY&HOLD" else 1.2)
                ax[1].plot(e.index, (e / e.cummax() - 1) * 100, lw=2.2 if nm == "BUY&HOLD" else 1.2)
            ax[0].set_yscale("log"); ax[0].set_title("Equity (log)"); ax[0].legend(fontsize=8)
            ax[1].set_title("Drawdown (%)")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "btc_mdd_equity_drawdown.png"), dpi=120)
            print(f"\nGrafik disimpan: {os.path.join(outdir, 'btc_mdd_equity_drawdown.png')}")
        except Exception as e:
            print(f"(grafik dilewati: {e})")
        print(f"Disimpan di {outdir}")
    return res, rets, (bh_cagr, bh_mdd)


def main():
    O, C = load_btc()
    main_analysis(O, C, outdir=RESULT_DIR)


if __name__ == "__main__":
    main()