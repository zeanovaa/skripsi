# Install Library
import os
os.system("pip install -q shap pmdarima")

# Import Library
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import pmdarima as pm
import shap
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import XGBRegressor

sns.set_theme(style="whitegrid")
os.makedirs("gambar", exist_ok=True)

# Konfigurasi
MAP = {"D102265TIC_PV": "X1", "D103260PIC_PV": "X2", "D101460TIC": "X3", "D101264FTS": "X4",
       "D101264FTSCL": "X5", "D101463PIC_PV": "X6", "GAS_MMBTU": "Y"}
ORDER = ["X1", "X2", "X4", "X3", "X5", "X6", "Y"]
XC = ORDER[:-1]
XS = ["X1", "X2", "X3", "X4", "X5", "X6"]
NAMA = {"X1": "Temperature Inlet Chamber", "X2": "Chamber Pressure", "X3": "Temperature Preheater",
        "X4": "Feed Density", "X5": "Flow Feed", "X6": "Nozzle Pressure", "Y": "Total Konsumsi Gas"}
SAT = {"X1": "°C", "X2": "mbar", "X3": "°C", "X4": "kg/L", "X5": "kg/h", "X6": "bar", "Y": "MMBtu"}
LIMIT = {"X1": (160, 195), "X2": (-1.0, -0.2), "X3": (69, 77), "X4": (1.11, 1.17),
         "X5": (3300, 4900), "X6": (145, 280)}
GRID = {"n_estimators": [250, 350, 450], "max_depth": [3, 4, 5], "learning_rate": [0.03, 0.08, 0.2],
        "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9], "gamma": [0.1, 0.5],
        "reg_lambda": [1, 6, 9], "reg_alpha": [0.3, 2]}
FUTURE = {"2026-03-04": "CKR", "2026-03-05": "CMP", "2026-03-06": "OFF", "2026-03-07": "CKR"}
LAGS, WINDOWS, MIN_HIST, HORIZON, SEED = (1, 2, 3, 4, 168, 699), (8, 24), 699, 24, 42
BIRU, MERAH = "#1f5a99", "#c24b43"
DIST = {"X1": "Temperatur Inlet Chamber", "X2": "Chamber pressure", "X3": "Temperatur Preheater",
        "X4": "Feed density", "X5": "Flow feed", "X6": "Nozzle pressure", "Y": "Total Konsumsi Gas"}

def tampil(fig, nama):
    fig.tight_layout()
    fig.savefig(f"gambar/{nama}.png", dpi=300, bbox_inches="tight")
    plt.show()

def lab(x):
    if x == "Y":
        return "Total Konsumsi Gas (MMBtu)"
    return "$\\mathit{" + NAMA[x].replace(" ", "\\ ") + "}$ (" + SAT[x] + ")"

def hist(ax, d, x, warna):
    sns.histplot(d[x], kde=True, color=warna, edgecolor="black", stat="count", ax=ax)
    ax.set_xlabel(DIST[x], fontsize=20)
    ax.set_ylabel("Frekuensi", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, linestyle="--", alpha=0.6)

def rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))

def smape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    d = np.abs(a) + np.abs(p)
    return float(np.mean(np.divide(2 * np.abs(a - p), d, out=np.zeros_like(d), where=d > 1e-12)) * 100)

# Muat Data
raw = pd.read_csv("18mar-3mar_notcap_jam.csv").rename(columns=MAP)[["PRODUK", *ORDER]]
raw["date"] = pd.date_range("2025-03-18 02:00:00", periods=len(raw), freq="h")
raw_nc = raw.copy()
TRAIN_END = int(len(raw) * 0.8)

# Capping
on = raw.PRODUK.ne("OFF")
tab41 = pd.DataFrame([{"Variabel": x, "< Batas Bawah": int((raw.loc[on, x] < lo).sum()),
                       "> Batas Atas": int((raw.loc[on, x] > hi).sum())} for x, (lo, hi) in LIMIT.items()])
tab41["Total"] = tab41["< Batas Bawah"] + tab41["> Batas Atas"]
tab41["Persentase (%)"] = (tab41.Total / on.sum() * 100).round(2)
print(tab41.to_string(index=False))
for x, (lo, hi) in LIMIT.items():
    raw.loc[on, x] = raw.loc[on, x].clip(lo, hi)

# Distribusi Jenis Produk
tab32 = raw.PRODUK.value_counts().to_frame("Jumlah")
tab32["Persentase (%)"] = (tab32.Jumlah / len(raw) * 100).round(2)
print(tab32)

# Statistik Deskriptif
d = raw[XS + ["Y"]]
tab43 = pd.DataFrame({"Mean": d.mean(), "Variance": d.var(), "Min": d.min(), "Q1": d.quantile(.25),
                      "Median": d.median(), "Q3": d.quantile(.75), "Max": d.max()})
print(tab43.round(2))

# Plot Konsumsi Gas Harian
h = raw.set_index("date").Y["2026-01-01":].resample("D").sum()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(h.index, h.values, color="tab:blue", lw=1)
ax.set_xlabel("Tanggal", fontsize=12)
ax.set_ylabel("Total Konsumsi Gas (MMBtu)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)
tampil(fig, "gambar_1_2_konsumsi_harian")

# Plot Distribusi Variabel
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
hist(ax[0], raw_nc, "X3", "skyblue")
hist(ax[1], raw_nc, "Y", "salmon")
tampil(fig, "gambar_1_3_distribusi")
for i in range(0, 6, 2):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for a, x in zip(ax, XS[i:i + 2]):
        hist(a, raw, x, "skyblue")
    tampil(fig, f"lampiran_2_distribusi_{XS[i]}_{XS[i + 1]}")

# Plot Deret Waktu
for x in XS + ["Y"]:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw.date, raw[x], color="tab:blue", lw=1)
    ax.set_xlabel("Bulan", fontsize=12)
    ax.set_ylabel(lab(x), fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    tampil(fig, f"deret_waktu_{x}")

# Plot ACF dan PACF
for x in ["Y"] + XS:
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(raw[x], lags=336, ax=ax, alpha=0.05, title=f"ACF {NAMA[x]}", markersize=0)
    ax.set_xlabel("Lag (jam)")
    tampil(fig, f"acf_{x}")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(raw[x], lags=336, ax=ax, alpha=0.05, method="ywm", title=f"PACF {NAMA[x]}", markersize=0)
    ax.set_xlabel("Lag (jam)")
    tampil(fig, f"pacf_{x}")

# Uji Musiman Analisis Spektral
y = raw.Y.to_numpy(float)
n = len(y)
ft = np.fft.rfft(y)[1:n // 2 + 1]
I = 2 * np.abs(ft) ** 2 / n
if n % 2 == 0:
    I[-1] = np.abs(ft[-1]) ** 2 / n
kk = np.arange(1, n // 2 + 1)
urut = np.argsort(I)[::-1]
baris = []
for r in range(2):
    i = urut[r]
    sub = I[urut[r:]]
    m = len(sub)
    baris.append({"k": int(kk[i]), "I": I[i], "P (jam)": n / kk[i],
                  "F hitung": (n - 3) * I[i] / (2 * (I.sum() - I[i])), "F tabel": stats.f.ppf(.95, 2, n - 3),
                  "T": I[i] / sub.sum(), "g alpha": 1 - (.05 / m) ** (1 / (m - 1))})
print(pd.DataFrame(baris).round(3).to_string(index=False))

# Uji RESET
rs = []
for x in XS:
    r = linear_reset(sm.OLS(raw_nc.Y, sm.add_constant(raw_nc[x])).fit(), power=3, test_type="fitted", use_f=True)
    rs.append({"Variabel": x, "F-statistic": round(float(r.statistic), 4),
               "p-value": "< 1e-300" if r.pvalue == 0 else f"{r.pvalue:.4e}",
               "Keputusan": "Tolak H0" if r.pvalue < .05 else "Terima H0"})
print(pd.DataFrame(rs).to_string(index=False))

# Pemodelan ARIMAX
arimax = pm.auto_arima(raw_nc.Y, X=raw_nc[XS], seasonal=False, stepwise=True, suppress_warnings=True)
print(arimax.summary())

# Fitur Deret Waktu
products = sorted(set(raw.PRODUK) | set(FUTURE.values()))
fut_dates = pd.date_range(raw.date.iloc[-1] + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
fut_prod = [FUTURE[t.strftime("%Y-%m-%d")] for t in fut_dates]

def fnames(t):
    n = ["hour_of_day", "day_of_week", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
         "shift_1", "shift_2", "shift_3"] + [f"PRODUK_{p}" for p in products]
    if t == "Y":
        n += [f"current_{c}" for c in XC]
    for c in (ORDER if t == "Y" else [t]):
        n += [f"{c}_lag_{l}" for l in LAGS]
        for w in WINDOWS:
            n += [f"{c}_mean_{w}", f"{c}_std_{w}"]
    return n

def make_features(t, date, product, hist_, cx=None):
    hr, dw = date.hour, date.dayofweek
    sh = 1 if 7 <= hr < 15 else 2 if 15 <= hr < 23 else 3
    v = [hr, dw, np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
         np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7),
         float(sh == 1), float(sh == 2), float(sh == 3)] + [float(product == p) for p in products]
    if t == "Y":
        v += [float(cx[c]) for c in XC]
    for c in (ORDER if t == "Y" else [t]):
        s = hist_[c]
        v += [float(s[-l]) for l in LAGS]
        for w in WINDOWS:
            seg = np.asarray(s[-w:], float)
            v += [float(seg.mean()), float(seg.std(ddof=1))]
    return v

H = {c: raw[c].to_numpy() for c in ORDER}

def matrix(t):
    rows = [make_features(t, raw.date.iloc[i], raw.PRODUK.iloc[i], {c: v[:i] for c, v in H.items()},
                          {c: H[c][i] for c in XC} if t == "Y" else None) for i in range(MIN_HIST, len(raw))]
    return pd.DataFrame(rows, columns=fnames(t), index=range(MIN_HIST, len(raw))), raw[t].iloc[MIN_HIST:]

MAT = {t: matrix(t) for t in ORDER}
print({t: MAT[t][0].shape[1] for t in ORDER})

# Pembagian Data
NTR = int((MAT["Y"][0].index < TRAIN_END).sum())
print(NTR, len(raw) - TRAIN_END)
for x in XS + ["Y"]:
    tr, te = raw.iloc[MIN_HIST:TRAIN_END], raw.iloc[TRAIN_END:]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(tr.date, tr[x], color="tab:blue", lw=0.8, label="Train")
    ax.plot(te.date, te[x], color="tab:red", lw=0.8, label="Test")
    ax.set_xlabel("Bulan")
    ax.set_ylabel(f"{NAMA[x]} ({SAT[x]})", fontstyle="italic")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.3, ls="--")
    ax.legend(loc="lower left")
    tampil(fig, f"split_{x}")

# Plot TimeSeriesSplit
fig, ax = plt.subplots(figsize=(10, 3.8))
for i, (tr, va) in enumerate(TimeSeriesSplit(n_splits=7, test_size=HORIZON).split(np.arange(NTR)), 1):
    ax.barh(i, len(tr), color="tab:blue", height=0.6, label="Train" if i == 1 else None)
    ax.barh(i, len(va), left=va[0], color="tab:red", height=0.6, label="Validasi" if i == 1 else None)
    ax.text(len(tr) - 3, i, f"{len(tr)}", ha="right", va="center", color="white", fontsize=8)
ax.set_xlim(5840, 6040)
ax.set_xlabel("Indeks observasi (jam)")
ax.set_ylabel("Fold")
ax.set_yticks(range(1, 8))
ax.invert_yaxis()
ax.grid(alpha=0.3, ls="--")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
tampil(fig, "fold_7")

# Grid Search
def model_for(p):
    return XGBRegressor(**p, objective="reg:squarederror", tree_method="hist", n_jobs=1,
                        random_state=SEED, verbosity=0)

def cv_rmse(t, p):
    f, l = MAT[t]
    fx, ly = f[f.index < TRAIN_END], l[l.index < TRAIN_END]
    ob, pr = [], []
    for a, b in TimeSeriesSplit(n_splits=7, test_size=HORIZON).split(fx):
        mdl = model_for(p).fit(fx.iloc[a], ly.iloc[a])
        ob += list(ly.iloc[b])
        pr += list(mdl.predict(fx.iloc[b]))
    return rmse(ob, pr)

CAND = [dict(zip(GRID, c)) for c in itertools.product(*GRID.values())]
RES, BEST = {}, {}
for t in ORDER:
    RES[t] = np.array(Parallel(n_jobs=-1, backend="threading")(delayed(cv_rmse)(t, p) for p in CAND))
    BEST[t] = CAND[int(np.argmin(RES[t]))]
    print(t, len(CAND), RES[t].min())
hp = pd.DataFrame({t: {**BEST[t], "RMSE validasi": RES[t].min()} for t in ORDER})[XS + ["Y"]]
print(hp.to_string())
top3 = np.argsort(RES["Y"])[:3]
print(pd.DataFrame([CAND[i] for i in top3]).assign(rmse_validasi=RES["Y"][top3]).to_string(index=False))

# Model Train 80 Persen
TR = {t: model_for(BEST[t]).fit(MAT[t][0][MAT[t][0].index < TRAIN_END], MAT[t][1][MAT[t][1].index < TRAIN_END])
      for t in ORDER}

# Fungsi Forecast End-to-End
def at(i):
    if i < len(raw):
        return raw.date.iloc[i], raw.PRODUK.iloc[i]
    return fut_dates[i - len(raw)], fut_prod[i - len(raw)]

def forecast(models, origin):
    fx = {}
    for t in XC:
        hh, nm, out = {t: raw[t].iloc[:origin].tolist()}, fnames(t), []
        for k in range(HORIZON):
            dt, pr = at(origin + k)
            v = float(models[t].predict(pd.DataFrame([make_features(t, dt, pr, hh)], columns=nm))[0])
            out.append(v)
            hh[t].append(v)
        fx[t] = out
    fx = pd.DataFrame(fx)
    hh = {c: raw[c].iloc[:origin].tolist() for c in ORDER}
    rows, yp = [], []
    for k in range(HORIZON):
        dt, pr = at(origin + k)
        r = make_features("Y", dt, pr, hh, fx.iloc[k])
        yp.append(float(models["Y"].predict(pd.DataFrame([r], columns=fnames("Y")))[0]))
        rows.append(r)
        for c in XC:
            hh[c].append(fx.at[k, c])
        hh["Y"].append(yp[-1])
    return fx, np.array(yp), pd.DataFrame(rows, columns=fnames("Y"))

# Evaluasi Rolling-Origin
ORIG = list(range(TRAIN_END, len(raw), HORIZON))

def run_origin(o):
    ms = {t: model_for(BEST[t]).fit(MAT[t][0][MAT[t][0].index < o], MAT[t][1][MAT[t][1].index < o])
          for t in ORDER}
    fx, yp, yf = forecast(ms, o)
    n = min(HORIZON, len(raw) - o)
    ex = shap.TreeExplainer(ms["Y"])
    sv = ex.shap_values(yf.iloc[:n])
    meta = pd.DataFrame({"date": raw.date.iloc[o:o + n].to_numpy(), "origin": raw.date.iloc[o],
                         "titik_asal": (o - TRAIN_END) // HORIZON + 1, "lead": np.arange(1, n + 1)})
    P = pd.concat([pd.DataFrame({"date": meta.date, "target": t, "actual": raw[t].iloc[o:o + n].to_numpy(),
                                 "prediction": (yp if t == "Y" else fx[t].to_numpy())[:n]}) for t in ORDER])
    return (P, meta.join(yf.iloc[:n].reset_index(drop=True)),
            meta.assign(base_value=float(np.ravel(ex.expected_value)[0])).join(pd.DataFrame(sv, columns=yf.columns)))

OUT = Parallel(n_jobs=-1, backend="threading")(delayed(run_origin)(o) for o in ORIG)
P = pd.concat([o[0] for o in OUT]).reset_index(drop=True)
FE = pd.concat([o[1] for o in OUT]).reset_index(drop=True)
SH = pd.concat([o[2] for o in OUT]).reset_index(drop=True)
print(len(ORIG), P.groupby("target").size().unique())

# Evaluasi Model Train dan Test
EV, PL = [], {}
for t in ORDER:
    f, l = MAT[t]
    a = f.index < TRAIN_END
    ytr, ptr = l[a].to_numpy(), TR[t].predict(f[a])
    dd = P[P.target == t].sort_values("date")
    PL[t] = (ytr, ptr, dd)
    EV.append({"Variabel": t, "RMSE Train": rmse(ytr, ptr), "RMSE Test": rmse(dd.actual, dd.prediction),
               "SMAPE Train (%)": smape(ytr, ptr), "SMAPE Test (%)": smape(dd.actual, dd.prediction)})
print(pd.DataFrame(EV).set_index("Variabel").loc[XS + ["Y"]].round(3).to_string())

# Plot Aktual dan Prediksi
for t in XS + ["Y"]:
    ytr, ptr, dd = PL[t]
    for nm, tgl, a, p in [("Train Data", raw.date.iloc[MIN_HIST:TRAIN_END], ytr, ptr),
                          ("Test Data", dd.date, dd.actual, dd.prediction)]:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(tgl, a, color=BIRU, lw=1, label="Aktual")
        ax.plot(tgl, p, color=MERAH, lw=1, label="Prediksi")
        ax.set_title(nm, fontsize=14)
        ax.set_xlabel("Bulan")
        ax.set_ylabel(f"{NAMA[t]} ({SAT[t]})", fontstyle="italic")
        ax.legend(loc="lower left")
        tampil(fig, f"aktual_prediksi_{t}_{nm.split()[0].lower()}")

# Analisis Nilai Nol dan Lonjakan X2
for x in ["X4", "X5"]:
    print(x, "nol train:", int((raw[x].iloc[MIN_HIST:TRAIN_END] == 0).sum()),
          "nol test:", int((raw[x].iloc[TRAIN_END:] == 0).sum()))
dd = PL["X2"][2]
e = dd.actual.to_numpy() - dd.prediction.to_numpy()
den = np.abs(dd.actual.to_numpy()) + np.abs(dd.prediction.to_numpy())
sm_i = np.divide(200 * np.abs(e), den, out=np.zeros_like(den), where=den > 1e-12)
mk = np.abs(e) > 2
print("jam lonjakan:", int(mk.sum()))
print("kontribusi SMAPE (poin %):", sm_i[mk].sum() / len(dd))
print("kontribusi kuadrat kesalahan:", (e[mk] ** 2).sum() / (e ** 2).sum())
print("proporsi SMAPE > 100%:", (sm_i > 100).mean())
print("proporsi beda tanda:", (np.sign(dd.actual.to_numpy()) != np.sign(dd.prediction.to_numpy())).mean())

# SHAP Global
cols = fnames("Y")
F, S = FE[cols], SH[cols]
shap.summary_plot(S.values, F, max_display=10, show=False)
plt.xlabel("Nilai SHAP (MMBtu)")
tampil(plt.gcf(), "shap_beeswarm")

def grup(c):
    for x in XC:
        if c == f"current_{x}" or c.startswith(x + "_"):
            return x
    if c.startswith("Y_"):
        return "Lag dan rolling window Y"
    if c.startswith("PRODUK_"):
        return "X7"
    return "Fitur waktu"

g = pd.Series({c: grup(c) for c in cols})
G = S.T.groupby(g).sum().T
tab412 = G.abs().mean().round(3).to_frame("Rata-rata |SHAP| (MMBtu)")
tab412["Jumlah fitur"] = g.value_counts()
print(tab412.loc[XS + ["X7", "Lag dan rolling window Y", "Fitur waktu"]])
print(S.abs().mean().nlargest(5).round(3).to_frame("Rata-rata |SHAP| (MMBtu)"))

# SHAP Lokal
ix = FE.index[FE.date == pd.Timestamp("2026-03-02 07:00:00")][0]
r = pd.DataFrame({"Fitur": cols, "Nilai Fitur": F.iloc[ix].to_numpy(), "SHAP": S.iloc[ix].to_numpy()})
top = r.loc[r.SHAP.abs().nlargest(5).index]
tab414 = pd.concat([top, pd.DataFrame([{"Fitur": f"Fitur lainnya ({len(r) - 5} fitur)",
                                        "SHAP": r.drop(top.index).SHAP.sum()}])]).round(3)
print("Titik asal ke-", FE.titik_asal[ix], "lead", FE.lead[ix], "origin", FE.origin[ix])
print("Base value:", round(SH.base_value[ix], 3), "Prediksi:", round(SH.base_value[ix] + r.SHAP.sum(), 3))
print(tab414.to_string(index=False))

# Retrain 100 Persen dan Real Forecast
FULL = {t: model_for(BEST[t]).fit(*MAT[t]) for t in ORDER}
fx, fy, _ = forecast(FULL, len(raw))
fx.index = fut_dates.strftime("%H:%M")
print(fx[XS].round(2).to_string())
print(pd.Series(fy, index=fx.index, name="Y").round(2).to_string())
print(pd.Series(fy).agg(["min", "max", "mean", "sum"]).round(2))

# Plot Real Forecast
a = raw.iloc[-7 * 24:]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(a.date, a.Y, color=BIRU, lw=1, label="Aktual")
ax.plot([raw.date.iloc[-1], *fut_dates], [raw.Y.iloc[-1], *fy], color=MERAH, lw=1.2, label="Forecast 24 jam")
ax.set(xlabel="Tanggal", ylabel="Total Konsumsi Gas (MMBtu)")
ax.legend(loc="lower left")
tampil(fig, "real_forecast_zoom")
