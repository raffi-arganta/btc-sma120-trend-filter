"""
Unduh kline harian BTCUSDT SPOT (2017-08 s/d END_MONTH, default 2026-09) dari Binance Vision dan simpan sebagai CSV BERHEADER,
supaya bisa dipakai btc_mdd_reduction.py lewat BTC_SPOT_DIR.

Pakai:  python download_btc_spot.py [folder_tujuan]
Default folder: <TF_BASE_DIR>/Dataset/btc_spot_1d
Lalu:   set BTC_SPOT_DIR=<folder_tujuan>        (cmd)
        $env:BTC_SPOT_DIR="<folder_tujuan>"     (PowerShell)
        python btc_mdd_reduction.py
"""
import io, os, sys, zipfile, urllib.request, urllib.error

BASE = os.environ.get("TF_BASE_DIR", r"D:\Quant_ML_Project\crypto_dashboard\mean reversion\fixed")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "Dataset", "btc_spot_1d")
URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-{y}-{m:02d}.zip"
END_MONTH = (2026, 9)
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def convert(zbytes):
    """Isi zip -> teks CSV berheader. Menangani file dengan/tanpa header."""
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    text = z.read(z.namelist()[0]).decode("utf-8").strip().splitlines()
    if text and not text[0].split(",")[0].strip().isdigit():
        text = text[1:]  # buang header bawaan
    out = []
    for line in text:
        p = line.split(",")
        for i in (0, 6):  # sejak 2025 timestamp spot bisa dalam mikrodetik -> jadikan milidetik
            if len(p[i].strip()) >= 16:
                p[i] = str(int(p[i]) // 1000)
        out.append(",".join(p))
    return ",".join(COLS) + "\n" + "\n".join(out) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = miss = 0
    for y in range(2017, END_MONTH[0] + 1):
        for m in range(1, 13):
            if (y, m) < (2017, 8) or (y, m) > END_MONTH:
                continue
            path = os.path.join(OUT, f"BTCUSDT-1d-{y}-{m:02d}.csv")
            if os.path.exists(path):
                ok += 1
                continue
            try:
                data = urllib.request.urlopen(URL.format(y=y, m=m), timeout=60).read()
                open(path, "w", newline="").write(convert(data))
                ok += 1
            except urllib.error.HTTPError as e:
                print(f"gagal {y}-{m:02d}: HTTP {e.code}")
                miss += 1
    print(f"Selesai: {ok} bulan tersimpan, {miss} gagal. Folder: {OUT}")
    if miss:
        print("Jalankan ulang untuk mencoba bulan yang gagal.")


if __name__ == "__main__":
    main()