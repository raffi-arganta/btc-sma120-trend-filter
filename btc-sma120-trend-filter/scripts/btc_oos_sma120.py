"""
UJI OOS SATU KALI: BTC SPOT, SMA120 long-only (di atas SMA = pegang BTC, di bawah = kas).

KRITERIA DIKUNCI SEBELUM MELIHAT HASIL (jangan diubah setelah run):
  Periode OOS: 2024-01-01 s/d data terakhir. Sinyal dihitung dengan seluruh riwayat sebelumnya (tanpa lookahead),
  eksekusi open t+1, compounding, biaya 0.10% per sisi (fee+slippage); 0.20% dan 0.30% sebagai stres.
  1. Calmar OOS SMA120 > Calmar buy&hold OOS.
  2. Calmar OOS SMA120 > Calmar alokasi statis dengan exposure rata-rata yang sama (timing menambah nilai).
  3. |MDD| SMA120 <= 70% |MDD| buy&hold OOS.
  4. CAGR OOS > 0 pada biaya 0.20% per sisi.
  5. Robust: dari tetangga L = 100,110,120,130,140, minimal 4 dari 5 lolos kriteria 1 DAN 3.
  LULUS jika 1-5 semuanya lolos.
ATURAN: hasil apa pun diterima. Kalau GAGAL, jangan ganti L, jangan tambah filter, jangan jalankan ulang
  dengan data/aturan lain. Skrip menulis file kunci setelah run pertama dan menolak run ulang
  (kecuali OOS_FORCE=1, yang berarti OOS sudah tidak bersih).
Keterbatasan: OOS ~2.75 tahun dan sangat sedikit episode bear; LULUS = konsisten dengan sampel lama, bukan bukti.
"""
import os, sys, datetime
import numpy as np
import pandas as pd
import explore_behavior as eb
import btc_mdd_reduction as bm

OOS_START = pd.Timestamp("2024-01-01")
L_MAIN, L_NEIGH = 120, (100, 110, 120, 130, 140)
FEES = (0.0010, 0.0020, 0.0030)
RESULT_DIR = os.path.join(eb.BASE_DIR, "Results", "btc_oos_sma120")
LOCK = os.path.join(RESULT_DIR, "oos_lock.txt")


def stats(r, bh_r):
    c, m = bm.cagr_mdd(r)
    up, dn = bh_r > 0, bh_r < 0
    return dict(CAGR=c * 100, MDD=m * 100, Calmar=c / abs(m) if m < 0 else np.nan,
                up_cap=r[up].mean() / bh_r[up].mean() * 100, down_cap=r[dn].mean() / bh_r[dn].mean() * 100)


def run_oos(O, C, outdir=None, write_lock=True):
    R = (O.shift(-1) / O - 1).iloc[:-1]
    C, O = C.loc[R.index], O.loc[R.index]
    cash_d = (1 + bm.CASH_ANN) ** (1 / 365) - 1
    gaps = C.index.to_series().diff().dropna()
    print(f"Data: {C.index[0].date()} s/d {C.index[-1].date()} | hari tidak kontinu: {int((gaps > pd.Timedelta(days=1)).sum())}")
    if C.index[-1] < OOS_START + pd.Timedelta(days=200):
        raise SystemExit("Data OOS terlalu pendek. Jalankan download_btc_spot.py (sampai 2026-09) lalu ulangi.")

    # sanity: reproduksi in-sample SMA120 (harus mendekati Calmar 2.34, CAGR 75.4%, MDD -32.3% jika data sama)
    is_start = C.index[0] + pd.Timedelta(days=bm.WARMUP)
    w120 = bm.sma_f(C, L_MAIN)
    R_is = R[R.index < OOS_START]
    r_is, _, _ = bm.evaluate(w120.loc[R_is.index], R_is, is_start, FEES[0], cash_d)
    c, m = bm.cagr_mdd(r_is)
    print(f"Sanity in-sample SMA{L_MAIN}: CAGR={c*100:.1f}% MDD={m*100:.1f}% Calmar={c/abs(m):.2f} (acuan sebelumnya 75.4% / -32.3% / 2.34)")

    bh, _, _ = bm.evaluate(pd.Series(1.0, index=C.index), R, OOS_START, FEES[0], cash_d)
    bh_s = stats(bh, bh)
    print(f"\nOOS {bh.index[0].date()} s/d {bh.index[-1].date()} ({len(bh)} hari)")
    print(f"BUY&HOLD OOS: CAGR={bh_s['CAGR']:.1f}% MDD={bh_s['MDD']:.1f}% Calmar={bh_s['Calmar']:.2f}")

    rows, res = [], {}
    for L in L_NEIGH:
        w = bm.sma_f(C, L)
        r, Ws, dW = bm.evaluate(w, R, OOS_START, FEES[0], cash_d)
        s = stats(r, bh)
        avg = Ws.mean()
        sc, sm = bm.cagr_mdd(avg * R[OOS_START:] + (1 - avg) * cash_d)
        cal_s = sc / abs(sm) if sm < 0 else np.nan
        cag20 = bm.cagr_mdd(bm.evaluate(w, R, OOS_START, FEES[1], cash_d)[0])[0] * 100
        cag30 = bm.cagr_mdd(bm.evaluate(w, R, OOS_START, FEES[2], cash_d)[0])[0] * 100
        k1 = s["Calmar"] > bh_s["Calmar"]
        k2 = s["Calmar"] > cal_s
        k3 = abs(s["MDD"]) <= 0.7 * abs(bh_s["MDD"])
        k4 = cag20 > 0
        rows.append(dict(L=L, CAGR=s["CAGR"], MDD=s["MDD"], Calmar=s["Calmar"], exposure=avg * 100,
                         turn_yr=dW.sum() / (len(r) / 365), up_cap=s["up_cap"], down_cap=s["down_cap"],
                         Calmar_static=cal_s, CAGR_f20=cag20, CAGR_f30=cag30, k1=k1, k2=k2, k3=k3, k4=k4))
        res[L] = (r, w)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== OOS: SMA120 dan tetangga (CAGR & MDD dalam %; k1..k4 = kriteria 1-4) ===")
    print(tab.round(2).to_string(index=False))

    main = tab[tab.L == L_MAIN].iloc[0]
    neigh_ok = int(((tab.k1) & (tab.k3)).sum())
    crit = {
        "1. Calmar > buy&hold": bool(main.k1),
        "2. Calmar > statis (exposure sama)": bool(main.k2),
        "3. |MDD| <= 70% MDD buy&hold": bool(main.k3),
        "4. CAGR > 0 pada fee 0.20%": bool(main.k4),
        f"5. robust: {neigh_ok}/5 tetangga lolos k1&k3 (butuh >=4)": bool(neigh_ok >= 4),
    }
    print("\n=== KRITERIA (dikunci) ===")
    for k, ok in crit.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    verdict = "LULUS" if all(crit.values()) else "GAGAL"
    print("  =>", verdict)

    r120, w = res[L_MAIN]
    print("\n=== PER TAHUN KALENDER (return %) ===")
    for y in sorted(set(r120.index.year)):
        a = r120[r120.index.year == y]
        b = bh[bh.index.year == y]
        print(f"  {y}: SMA{L_MAIN}={((1+a).prod()-1)*100:7.1f}% (MDD {bm.cagr_mdd(a)[1]*100:6.1f}%) | "
              f"BTC={((1+b).prod()-1)*100:7.1f}% (MDD {bm.cagr_mdd(b)[1]*100:6.1f}%) hari={len(a)}")
    eq_bh = (1 + bh).cumprod()
    td = (eq_bh / eq_bh.cummax() - 1).idxmin()
    pk = eq_bh[:td].idxmax()
    eq = (1 + r120).cumprod()
    print(f"  Drawdown terbesar BTC di OOS: {pk.date()} -> {td.date()} = {(eq_bh[td]/eq_bh[pk]-1)*100:.1f}%; "
          f"SMA{L_MAIN} pada jendela itu: {(eq[td]/eq[pk]-1)*100:.1f}%")

    ws = w.shift(1).fillna(0.0)[OOS_START:]
    ch = ws.diff().fillna(ws.iloc[0])
    sw = ch[ch != 0]
    print(f"\n=== PERGANTIAN POSISI SMA{L_MAIN} di OOS (tanggal eksekusi = open hari itu) ===")
    ent = None
    for d, v in sw.items():
        px = O.get(d, np.nan)
        if v > 0:
            ent = (d, px)
            print(f"  MASUK  {d.date()} @ {px:,.0f}")
        else:
            tr = (px / ent[1] - 1) * 100 if ent else np.nan
            print(f"  KELUAR {d.date()} @ {px:,.0f}  (trade {tr:+.1f}%)")
            ent = None
    if ent:
        print(f"  (posisi terbuka sejak {ent[0].date()})")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        tab.to_csv(os.path.join(outdir, "oos_table.csv"), index=False)
        pd.DataFrame({"SMA120": r120, "BUYHOLD": bh}).to_csv(os.path.join(outdir, "oos_daily_returns.csv"))
        if write_lock:
            open(LOCK, "w").write(f"OOS dijalankan {datetime.datetime.now():%Y-%m-%d %H:%M} | L={L_MAIN} | hasil={verdict}\n")
        print(f"\nDisimpan di {outdir}")
    return verdict


def main():
    if os.path.exists(LOCK) and os.environ.get("OOS_FORCE") != "1":
        print(open(LOCK).read())
        raise SystemExit("OOS sudah pernah dijalankan. Run ulang membuat OOS tidak bersih (OOS_FORCE=1 untuk memaksa).")
    eb.END = pd.Timestamp("2026-12-31")
    O, C = bm.load_btc()
    run_oos(O, C, outdir=RESULT_DIR)


if __name__ == "__main__":
    main()