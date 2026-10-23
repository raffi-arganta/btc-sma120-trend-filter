"""
EKSPLORASI PERILAKU CROSS-SECTIONAL (tanpa strategi, tanpa fee, tanpa sizing).

Pertanyaan: fitur apa (diukur di close hari t) yang memprediksi return relatif antar-koin
selama h hari ke depan (entry open t+1, open-to-open)?

Metode:
  - Universe point-in-time sama dengan kodemu (umur>=120 hari, vol 30h>=5jt USD).
  - Target: forward return h hari, di-winsorize 1/99% per hari, dikurangi median lintas koin
    (jadi murni cross-sectional / market-neutral).
  - IC = Spearman(rank fitur, rank target) per hari. t-stat dihitung dari sampel NON-OVERLAP
    (setiap h hari) supaya tidak menggelembung karena overlap.
  - Kuintil: rata-rata target per kuintil fitur; spread = Q5 - Q1 (bps per hari) + monotonisitas.
  - Discovery < 2022-07-01, validation >= 2022-07-01 s/d 2023-12-31. OOS 2024+ TIDAK disentuh.
  - Jumlah hipotesis = fitur x horizon. Ambang discovery = sqrt(2 ln H) (Bonferroni-ish).

FLAG "KANDIDAT" jika: |t_D| >= ambang, tanda IC sama di validation dengan |t_V| >= 1.65,
  dan tanda IC sama di >= 3 dari 4 tahun kalender, dan sama di bear & bull.
Hasil "tidak ada kandidat" adalah hasil valid.

Keterbatasan: koin yang delisting dan datanya berhenti hilang dari target (survivorship ringan).
"""
import glob, os, math
import numpy as np
import pandas as pd

BASE_DIR = os.environ.get("TF_BASE_DIR", r"D:\Quant_ML_Project\crypto_dashboard\mean reversion\fixed")
VISION = os.path.join(BASE_DIR, "Dataset", "binance_vision_all")
PERP_DIR = os.path.join(VISION, "klines_1h")
FUND_DIR = os.path.join(VISION, "funding")
RESULT_DIR = os.path.join(BASE_DIR, "Results", "behavior_explore")

END = pd.Timestamp("2023-12-31")
START_EVAL = pd.Timestamp("2020-07-01")
SPLIT = pd.Timestamp("2022-07-01")
MIN_AGE, MIN_VOL, MIN_NAMES = 120, 5_000_000.0, 20
HORIZONS = (1, 3, 7)

TIME_COLS = ["timestamp", "open_time", "calc_time", "fundingTime", "funding_time", "time", "datetime", "date"]
RATE_COLS = ["last_funding_rate", "fundingRate", "funding_rate", "rate"]


# ---------- loader (sama dengan skrip breakout) ----------
def pick(cols, options):
    low = {str(c).lower(): c for c in cols}
    for o in options:
        if o.lower() in low:
            return low[o.lower()]
    raise KeyError(f"kolom {options} tidak ada. Tersedia: {list(cols)}")


def to_time(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms" if float(s.iloc[0]) > 1e11 else "s")
    t = pd.to_datetime(s)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t


def find_files(d):
    files = glob.glob(os.path.join(d, "*.csv")) or glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
    out = {}
    for f in files:
        out.setdefault(os.path.basename(f).split("-")[0].split(".")[0].upper(), []).append(f)
    return out


def load_daily(files):
    parts = []
    for f in files:
        d = pd.read_csv(f)
        parts.append(pd.DataFrame({
            "t": to_time(d[pick(d.columns, TIME_COLS)]), "open": d[pick(d.columns, ["open"])],
            "high": d[pick(d.columns, ["high"])], "low": d[pick(d.columns, ["low"])],
            "close": d[pick(d.columns, ["close"])],
            "qv": d[pick(d.columns, ["quote_volume", "quote_asset_volume", "qv"])]}))
    k = pd.concat(parts).drop_duplicates("t").set_index("t").sort_index()
    k = k[k.index < END + pd.Timedelta(days=1)]
    dd = k.resample("D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "qv": "sum"})
    n = k["open"].resample("D").count()
    return dd[n >= 20]


def load_funding_daily(files):
    parts = []
    for f in files:
        d = pd.read_csv(f)
        parts.append(pd.DataFrame({"t": to_time(d[pick(d.columns, TIME_COLS)]),
                                   "r": d[pick(d.columns, RATE_COLS)].astype(float)}))
    d = pd.concat(parts).drop_duplicates("t").sort_values("t")
    return d.set_index("t")["r"].resample("D").sum()


def build():
    perp, fund = find_files(PERP_DIR), find_files(FUND_DIR)
    print(f"File perp: {len(perp)}, funding: {len(fund)}")
    if not perp:
        raise SystemExit("Folder perp kosong. Cek path.")
    D, F = {}, {}
    for sym, files in perp.items():
        try:
            d = load_daily(files)
        except Exception as e:
            print(f"lewati {sym}: {e}")
            continue
        if len(d) < MIN_AGE + 60:
            continue
        D[sym] = d
        if sym in fund:
            try:
                F[sym] = load_funding_daily(fund[sym])
            except Exception:
                pass
    if "BTCUSDT" not in D:
        raise SystemExit("BTCUSDT tidak ada.")
    idx = pd.date_range(min(d.index[0] for d in D.values()), END)
    P = {c: pd.DataFrame({s: d[c] for s, d in D.items()}).reindex(idx) for c in ("open", "high", "low", "close", "qv")}
    Fd = pd.DataFrame(F).reindex(idx).reindex(columns=P["open"].columns).fillna(0.0)
    print(f"Simbol dipakai: {len(D)} | hari: {len(idx)}")
    return P, Fd


# ---------- fitur (semua hanya memakai info sampai close hari t) ----------
def make_features(P, Fd):
    O, H, L, C, V = P["open"], P["high"], P["low"], P["close"], P["qv"]
    r1 = C.pct_change(fill_method=None)
    btc = r1["BTCUSDT"]
    rng = (H - L) / C
    v30 = V.rolling(30, min_periods=20).mean()
    f = {
        "ret1 (reversal 1d)": r1,
        "ret3": C / C.shift(3) - 1,
        "ret7": C / C.shift(7) - 1,
        "ret14": C / C.shift(14) - 1,
        "ret28": C / C.shift(28) - 1,
        "mom_28_skip7": C.shift(7) / C.shift(28) - 1,
        "mom_90_skip7": C.shift(7) / C.shift(90) - 1,
        "vol_shock (vol/avg30)": V / v30.shift(1),
        "range_expansion": rng / rng.rolling(30, min_periods=20).mean().shift(1),
        "sigma30 (low-vol anomaly)": r1.rolling(30, min_periods=20).std(),
        "max30 (lottery)": r1.rolling(30, min_periods=20).max(),
        "min30": r1.rolling(30, min_periods=20).min(),
        "skew30": r1.rolling(30, min_periods=20).skew(),
        "dist_high55": C / H.rolling(55, min_periods=40).max() - 1,
        "dist_low55": C / L.rolling(55, min_periods=40).min() - 1,
        "close_location": (C - L) / (H - L).replace(0, np.nan),
        "fund3d": Fd.rolling(3).sum(),
        "fund7d": Fd.rolling(7).sum(),
        "size (log vol30)": np.log(v30.where(v30 > 0)),
        "beta_btc60": r1.rolling(60, min_periods=40).cov(btc) / btc.rolling(60, min_periods=40).var(),
    }
    return f


def make_targets(P):
    O = P["open"]
    out = {}
    for h in HORIZONS:
        t = O.shift(-(1 + h)) / O.shift(-1) - 1
        lo, hi = t.quantile(0.01, axis=1), t.quantile(0.99, axis=1)
        t = t.clip(lower=lo, upper=hi, axis=0)
        out[h] = t.sub(t.median(axis=1), axis=0)
    return out


def tstat(x, h):
    x = x.dropna().iloc[::h]
    if len(x) < 20 or x.std() == 0:
        return np.nan, len(x)
    return x.mean() / (x.std() / math.sqrt(len(x))), len(x)


def analyze(P, Fd, quiet=False):
    C, V = P["close"], P["qv"]
    elig = ((C.notna().cumsum() >= MIN_AGE) & (V.rolling(30, min_periods=20).mean() >= MIN_VOL) & C.notna())
    feats, targs = make_features(P, Fd), make_targets(P)
    sma = C["BTCUSDT"].rolling(200).mean()
    bear = (C["BTCUSDT"] < sma)

    nh = len(feats) * len(HORIZONS)
    thr = math.sqrt(2 * math.log(nh))
    if not quiet:
        print(f"\nHipotesis diuji: {nh} ({len(feats)} fitur x {len(HORIZONS)} horizon). Ambang t discovery = {thr:.2f}")

    rows = []
    for name, F in feats.items():
        for h in HORIZONS:
            T = targs[h]
            valid = elig & F.notna() & T.notna()
            enough = valid.sum(axis=1) >= MIN_NAMES
            fr = F.where(valid).rank(axis=1)
            tr = T.where(valid).rank(axis=1)
            ic = fr.corrwith(tr, axis=1).where(enough)
            ic = ic[ic.index >= START_EVAL]

            pct = F.where(valid).rank(axis=1, pct=True)
            qm = []
            for q in range(1, 6):
                m = (pct > (q - 1) / 5) & (pct <= q / 5)
                qm.append(T.where(m).mean(axis=1).where(enough))
            Q = pd.concat(qm, axis=1)
            Q.columns = range(1, 6)
            Q = Q[Q.index >= START_EVAL]
            spread = (Q[5] - Q[1])

            dD, dV = ic[ic.index < SPLIT], ic[ic.index >= SPLIT]
            tD, _ = tstat(dD, h)
            tV, _ = tstat(dV, h)
            b = bear.reindex(ic.index).fillna(False)
            icB, icU = ic[b].mean(), ic[~b].mean()
            yrs = [ic[(ic.index >= f"{y}-01-01") & (ic.index < f"{y+1}-01-01")].mean() for y in (2020, 2021, 2022, 2023)]
            sgn = np.sign(dD.mean())
            yrs_ok = sum(np.sign(y) == sgn for y in yrs if not np.isnan(y))
            mono = lambda s: pd.Series(Q.loc[s.index].mean().values).corr(pd.Series(range(1, 6)), method="spearman")
            sD, sV = spread[spread.index < SPLIT], spread[spread.index >= SPLIT]
            cand = bool(
                abs(tD) >= thr and np.sign(dV.mean()) == sgn and abs(tV) >= 1.65
                and yrs_ok >= 3 and np.sign(icB) == sgn and np.sign(icU) == sgn)
            rows.append(dict(feature=name, h=h, IC_D=dD.mean(), t_D=tD, IC_V=dV.mean(), t_V=tV,
                             IC_bear=icB, IC_bull=icU, yrs_same_sign=yrs_ok,
                             Q5mQ1_D_bps_day=sD.mean() * 1e4 / h, Q5mQ1_V_bps_day=sV.mean() * 1e4 / h,
                             mono_D=mono(sD), mono_V=mono(sV), KANDIDAT=cand))
    res = pd.DataFrame(rows)
    return res, thr


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    P, Fd = build()
    res, thr = analyze(P, Fd)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = res.copy()
    for c in ("IC_D", "IC_V", "IC_bear", "IC_bull"):
        show[c] = (show[c] * 100).round(2)  # IC dalam %
    show = show.round(2).sort_values("t_D", key=lambda s: -s.abs())
    print("\n=== SEMUA FITUR (IC dalam %, diurut |t discovery|; spread bps/hari BRUTO sebelum fee) ===")
    print(show.to_string(index=False))
    c = res[res.KANDIDAT]
    print(f"\n=== KANDIDAT: {len(c)} dari {len(res)} ===")
    if len(c):
        print(c[["feature", "h", "IC_D", "t_D", "IC_V", "t_V", "Q5mQ1_D_bps_day", "Q5mQ1_V_bps_day"]].round(3).to_string(index=False))
        print("\nLangkah berikut: pastikan spread bruto > biaya turnover (long-short kuintil, fee 0.10%/sisi),"
              "\ntulis alasan ekonominya, baru bangun strategi. Jangan sentuh OOS 2024+.")
    else:
        print("Tidak ada pola cross-sectional yang lolos. Ini hasil valid: perilaku ini tidak mengandung edge.")
    res.to_csv(os.path.join(RESULT_DIR, "behavior_ic_table.csv"), index=False)
    print(f"\nDisimpan di {RESULT_DIR}")


if __name__ == "__main__":
    main()