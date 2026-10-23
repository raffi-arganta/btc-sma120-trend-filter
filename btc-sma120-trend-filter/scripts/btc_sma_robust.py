"""
SMA100 -> PETA ROBUSTNESS (bukan mencari angka terbaik). Taruh satu folder dengan explore_behavior.py dan
btc_mdd_reduction.py. Data: pakai BTC_SPOT_DIR (spot 2017+) bila diset.

DESAIN (dikunci sebelum melihat hasil):
  Grid: SMA L = 80..160 (langkah 10) x 12 varian filter = 108 konfigurasi.
  Varian: SMA polos | +momentum 60/90 hari > 0 | +keluar saat lonjakan volatilitas (persentil vol30 dlm 365 hari
          > 80/90) | +hysteresis band 3%/5% | staged (100% bila SMA&momentum 60/90, 50% bila hanya SMA)
          | +volatility scaling (target 60/80% saat di atas SMA) | +cek mingguan.
  Eksekusi: sinyal di close t, posisi dari open t+1. Compounding. Biaya per sisi 0.10/0.20/0.30% (fee+slippage).
  Pemilihan L per varian: SMOOTHED Calmar = rata-rata Calmar di L-10, L, L+10 (titik tepi tidak dipilih).
  Ukuran robust: % USEFUL (|MDD|<=70% MDD BH dan Calmar>BH), % mengalahkan alokasi statis di exposure yang sama,
          % lolos di KEDUA paruh sampel (sebelum/sesudah 2021-01-01), Calmar terburuk di seluruh L, Calmar fee 0.30%.
  Filter dianggap MENAMBAH NILAI hanya jika smoothed Calmar naik >= 0.15 dibanding SMA polos DAN
          % lolos kedua paruh tidak lebih rendah. Bila tidak, filter itu hanya menambah kerumitan.
  Episode (jendela tetap dari kejadian pasar): 2018H2, Covid 2020, Mei 2021, Bear 2022.
Keterbatasan: 2 bear market, 108 konfigurasi (best-of-108 = optimistis), parameter filter dipilih saya.
OOS 2024+ tidak disentuh.
"""
import os, math
import numpy as np
import pandas as pd
import explore_behavior as eb
import btc_mdd_reduction as bm

LS = list(range(80, 161, 10))
FEES = (0.0010, 0.0020, 0.0030)
HALF = pd.Timestamp("2021-01-01")
EPISODES = {"ep_2018H2": ("2018-06-13", "2018-12-15"), "ep_Covid20": ("2020-02-13", "2020-03-16"),
            "ep_Mei21": ("2021-04-14", "2021-07-20"), "ep_Bear22": ("2021-11-08", "2022-11-21")}
GAIN_MIN = 0.15
RESULT_DIR = os.path.join(eb.BASE_DIR, "Results", "btc_sma_robust")


def build_families(C):
    vol = C.pct_change().rolling(30, min_periods=20).std()
    vol_ann = vol * math.sqrt(365)
    vrank = vol.rolling(365, min_periods=180).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    mom = {M: ((C / C.shift(M) - 1) > 0).astype(float) for M in (60, 90)}
    F = {"SMA polos": {L: bm.sma_f(C, L) for L in LS}}
    for M in (60, 90):
        F[f"+mom{M}"] = {L: bm.sma_f(C, L) * mom[M] for L in LS}
    for q in (0.80, 0.90):
        ok = (~(vrank > q).fillna(False)).astype(float)
        F[f"+volspike{int(q*100)}"] = {L: bm.sma_f(C, L) * ok for L in LS}
    for b in (0.03, 0.05):
        F[f"+hyst{int(b*100)}%"] = {L: bm.hyst_f(C, L, b) for L in LS}
    for M in (60, 90):
        F[f"staged_mom{M}"] = {L: bm.sma_f(C, L) * (0.5 + 0.5 * mom[M]) for L in LS}
    for t in (0.60, 0.80):
        F[f"+volscale{int(t*100)}"] = {L: bm.sma_f(C, L) * (t / vol_ann).clip(0, 1).fillna(0.0) for L in LS}
    F["+weekly"] = {L: bm.weekly_f(C, L) for L in LS}
    return F


def calmar(r):
    if len(r) < 60:
        return np.nan
    c, m = bm.cagr_mdd(r)
    return c / abs(m) if m < 0 else np.nan


def ep_ret(r, a, b):
    s = r[(r.index >= a) & (r.index <= b)]
    return ((1 + s).prod() - 1) * 100 if len(s) > 5 else np.nan


def analyze(O, C, outdir=None, quiet=False):
    start = C.index[0] + pd.Timedelta(days=bm.WARMUP)
    R = (O.shift(-1) / O - 1).iloc[:-1]
    C, O = C.loc[R.index], O.loc[R.index]
    cash_d = (1 + bm.CASH_ANN) ** (1 / 365) - 1
    fams = build_families(C)
    nconf = sum(len(v) for v in fams.values())

    bh, _, _ = bm.evaluate(pd.Series(1.0, index=C.index), R, start, FEES[0], cash_d)
    bh_c, bh_m = bm.cagr_mdd(bh)
    bh_cal = bh_c / abs(bh_m)
    bh_h1, bh_h2 = calmar(bh[bh.index < HALF]), calmar(bh[bh.index >= HALF])
    print(f"\nEvaluasi {bh.index[0].date()} s/d {bh.index[-1].date()} | konfigurasi: {nconf} (best-of-{nconf} akan optimistis)")
    print(f"BUY&HOLD: CAGR={bh_c*100:.1f}% MDD={bh_m*100:.1f}% Calmar={bh_cal:.2f} | Calmar paruh1={bh_h1:.2f} paruh2={bh_h2:.2f}")
    print("Episode BH: " + " | ".join(f"{k[3:]}={ep_ret(bh, *v):.0f}%" for k, v in EPISODES.items()))

    rows, rets = [], {}
    for fam, d in fams.items():
        for L, w in d.items():
            r, Ws, dW = bm.evaluate(w, R, start, FEES[0], cash_d)
            c, m = bm.cagr_mdd(r)
            cal = c / abs(m) if m < 0 else np.nan
            avg = Ws.mean()
            sc, sm = bm.cagr_mdd(avg * R[start:] + (1 - avg) * cash_d)
            cal_s = sc / abs(sm) if sm < 0 else np.nan
            row = dict(family=fam, L=L, CAGR=c * 100, MDD=m * 100, Calmar=cal, exposure=avg * 100,
                       turn_yr=dW.sum() / (len(r) / 365), Calmar_static=cal_s)
            for f in FEES[1:]:
                rf = bm.evaluate(w, R, start, f, cash_d)[0]
                cf, mf = bm.cagr_mdd(rf)
                row[f"Calmar_f{int(f*1e4)}"] = cf / abs(mf) if mf < 0 else np.nan
            row["Calmar_H1"], row["Calmar_H2"] = calmar(r[r.index < HALF]), calmar(r[r.index >= HALF])
            for k, (a, b) in EPISODES.items():
                row[k] = ep_ret(r, a, b)
            row["USEFUL"] = bool(abs(m) <= 0.7 * abs(bh_m) and cal > bh_cal)
            row["BEATS_STATIC"] = bool(cal > cal_s)
            row["BOTH_HALVES"] = bool(row["Calmar_H1"] > bh_h1 and row["Calmar_H2"] > bh_h2)
            rows.append(row)
            rets[(fam, L)] = r
    res = pd.DataFrame(rows)

    # peta Calmar & MDD
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    order = list(fams.keys())
    cal_map = res.pivot(index="L", columns="family", values="Calmar")[order]
    mdd_map = res.pivot(index="L", columns="family", values="MDD")[order]
    print("\n=== PETA CALMAR (baris L, kolom varian; BH =", f"{bh_cal:.2f})")
    print(cal_map.round(2).to_string())
    print("\n=== PETA MDD % (BH =", f"{bh_m*100:.1f})")
    print(mdd_map.round(1).to_string())

    # ringkasan per varian
    g = res.groupby("family", sort=False)
    sm = cal_map.rolling(3, center=True).mean()
    summ = pd.DataFrame({
        "Calmar_med": g.Calmar.median(), "Calmar_min": g.Calmar.min(), "MDD_med": g.MDD.median(),
        "CAGR_med": g.CAGR.median(), "expo_med": g.exposure.median(), "turn_yr": g.turn_yr.median(),
        "Calmar_f30_med": g.Calmar_f30.median(),
        "pct_USEFUL": g.USEFUL.mean() * 100, "pct_BEATS_STATIC": g.BEATS_STATIC.mean() * 100,
        "pct_BOTH_HALVES": g.BOTH_HALVES.mean() * 100,
        "best_L_smooth": sm.idxmax(), "smooth_Calmar": sm.max()})
    base_s = summ.loc["SMA polos", "smooth_Calmar"]
    base_h = summ.loc["SMA polos", "pct_BOTH_HALVES"]
    summ["delta_vs_polos"] = summ.smooth_Calmar - base_s
    summ["ADDS_VALUE"] = (summ.delta_vs_polos >= GAIN_MIN) & (summ.pct_BOTH_HALVES >= base_h)
    summ.loc["SMA polos", "ADDS_VALUE"] = False  # pembanding
    print("\n=== RINGKASAN PER VARIAN (robust = nilai stabil di banyak L, bukan satu titik) ===")
    print(summ.round(2).to_string())

    # detail di L terpilih (smoothed) per varian
    pick = res.merge(pd.DataFrame({"family": summ.index, "L": summ.best_L_smooth.values}), on=["family", "L"])
    cols = ["family", "L", "CAGR", "MDD", "Calmar", "exposure", "turn_yr", "Calmar_f20", "Calmar_f30",
            "Calmar_H1", "Calmar_H2"] + list(EPISODES) + ["BEATS_STATIC", "USEFUL"]
    print("\n=== DETAIL DI L TERPILIH (smoothed Calmar; bukan titik tertinggi mentah) ===")
    print(pick[cols].round(2).to_string(index=False))

    ok = summ[(summ.pct_USEFUL >= 70) & (summ.pct_BOTH_HALVES >= 60) & (summ.Calmar_f30_med > bh_cal)]
    print("\n=== ATURAN PRA-DEFINISI ===")
    print("Lolos (>=70% USEFUL, >=60% lolos kedua paruh, Calmar fee 0.30% > BH): "
          + (", ".join(ok.index) if len(ok) else "tidak ada"))
    add = summ[summ.ADDS_VALUE == True]
    print("Filter yang MENAMBAH nilai dibanding SMA polos: " + (", ".join(add.index) if len(add) else
          "tidak ada -> filter hanya menambah kerumitan, SMA polos cukup"))
    print(f"\nAda {nconf} konfigurasi. Anggap Calmar tertinggi optimistis. Bandingkan sebagai gambaran, bukan rekomendasi. "
          "OOS 2024+ jangan disentuh sampai satu aturan dikunci.")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        res.to_csv(os.path.join(outdir, "robust_all.csv"), index=False)
        summ.to_csv(os.path.join(outdir, "robust_family_summary.csv"))
        cal_map.to_csv(os.path.join(outdir, "robust_calmar_map.csv"))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            for fam in order:
                ax[0].plot(cal_map.index, cal_map[fam], lw=2.4 if fam == "SMA polos" else 1.0, label=fam)
                ax[1].plot(mdd_map.index, mdd_map[fam], lw=2.4 if fam == "SMA polos" else 1.0)
            ax[0].axhline(bh_cal, color="k", ls="--"); ax[0].set_title("Calmar vs L (garis putus = buy&hold)")
            ax[1].axhline(bh_m * 100, color="k", ls="--"); ax[1].set_title("MDD % vs L")
            ax[0].legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "robust_landscape.png"), dpi=120)
        except Exception as e:
            print(f"(grafik dilewati: {e})")
        print(f"Disimpan di {outdir}")
    return res, summ


def main():
    O, C = bm.load_btc()
    analyze(O, C, outdir=RESULT_DIR)


if __name__ == "__main__":
    main()