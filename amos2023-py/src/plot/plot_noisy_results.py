import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import data, colours
from tomography import psf
import numpy as np

from sklearn import linear_model

df = data.load_noise()
df0 = data.load_noise()
df1 = data.load_noise()

sns.set_theme(context="poster", style="whitegrid", font="serif", rc={"lines.linewidth": 1.25})

clr_dict = colours.palette()

r = psf.get_resolution(600e-9, 0.6, 6.5)

tgts = df["Target_Gamma"].unique()

for snr in df["SNR"].unique():
    if snr != 0:
        for b in df["Basis"].unique():
            df.loc[(df["SNR"] == snr) & (df["Basis"] == b), "RMSE"] = df.loc[(df["SNR"] == snr) & (df["Basis"] == b), "RMSE_RA"] # - df.loc[(df["SNR"] == snr) & (df["Basis"] == "Wavelets (Haar)"), "RMSE_RA"]
        df.loc[:, "Data:"] = "Noise"
        # for tgt in tgts:

        #     df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "RMSE"] = df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "RMSE_RA"] # / df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "RMSE_BA"] #* 100
            # df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "FWHM"] = (df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "FWHM_RA"] / df.loc[(df["SNR"] == 0) & (df["Target_Gamma"] == tgt), "FWHM_RA"]) * 100
            # df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "MSSIM"] = (df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "MSSIM_RA"] / df.loc[(df["SNR"] == 0) & (df["Target_Gamma"] == tgt), "MSSIM_RA"]) * 100
            # df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "Strehl"] = (df.loc[(df["SNR"] == snr) & (df["Target_Gamma"] == tgt), "Strehl_RB"] / df.loc[(df["SNR"] == 0) & (df["Target_Gamma"] == tgt), "Strehl_RB"]) * 100



df0.loc[:, "RMSE"] = df0.loc[:, "RMSE_BA"]
df0.loc[:, "Data:"] = "Ground Truth"

for b in df1["Basis"].unique():
    df1.loc[(df1["N_Coeffs"] == 16) & (df1["Basis"] == b), "RMSE"] = df1.loc[(df["SNR"] == 0) & (df1["N_Coeffs"] == 16) & (df1["Basis"] == b), "RMSE_RA"].mean()
    df1.loc[(df1["N_Coeffs"] == 64) & (df1["Basis"] == b), "RMSE"] = df1.loc[(df["SNR"] == 0) & (df1["N_Coeffs"] == 64) & (df1["Basis"] == b), "RMSE_RA"].mean()
df1.loc[:, "Data:"] = "No Noise"
# ast1_0 = "Single Star"
# ast3_15 = "3 stars, 15\" radius"
# ast5_Asym = "5 stars asymmetric"
# df.loc[df["Asterism"] == "1_0", "Asterism"] = ast1_0
# df.loc[df["Asterism"] == "3_15", "Asterism"] = ast3_15
# df.loc[df["Asterism"] == "5_Asym", "Asterism"] = ast5_Asym

df = pd.concat([df, df0, df1])

df = df.loc[df["SNR"] != 0, :]
# df = df.loc[df["Target_Gamma"] != 0, :]
# df = df.loc[(df["Target_Gamma"] == 0) | (df["Target_Gamma"] == 30) | (df["Target_Gamma"] == 15), :]
# df = df.loc[df["Asterism"] == ast3_15, :]

n_coeffs = "Number of Modes:"
gamma = "Angle"
df.rename(columns={"N_Coeffs": n_coeffs}, inplace=True)
df.rename(columns={"Target_Gamma": gamma}, inplace=True)
df.loc[df["SNR"] == 0, "SNR"] = "No Noise"

hue_order = ["Noise", "No Noise", "Ground Truth"]
col_order = ["Zernike", "DCT", "DST", "Wavelets (Haar)", "Wavelets (db2)"]

clr_dict[1], clr_dict[2] = clr_dict[5], clr_dict[1]

g = sns.relplot(
    data=df, kind='line',
    y="RMSE", x="SNR", col="Basis",
    hue="Data:", style=n_coeffs,
    palette=clr_dict, errorbar=None,
    col_order=col_order, hue_order=hue_order,
    col_wrap=3,  facet_kws=dict(legend_out=False)
)

sns.move_legend(g, "upper left", bbox_to_anchor=(.72, .48), frameon=True)

(
    g.set_axis_labels("SNR (dB)", "RMSE (radians)")
    .set(xscale='log')
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.invert_xaxis()

plt.show()
quit()

def r_square(y_true, y_pred):
    SSres = np.sum((y_true - y_pred) ** 2)
    SStot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (SSres / SStot)

for b in df["Basis"].unique():
    # for g in df[gamma].unique():
    for c in df[n_coeffs].unique():
        g = 0
        # x = np.array(df.loc[(df[gamma] == g) & (df["Basis"] == b) & (df[n_coeffs] == c), "SNR"], dtype=np.float64)
        # y = np.array(df.loc[(df[gamma] == g) & (df["Basis"] == b) & (df[n_coeffs] == c), "RMSE"])
        x = np.array(df.loc[(df["Basis"] == b) & (df[n_coeffs] == c), "SNR"], dtype=np.float64)
        y = np.array(df.loc[(df["Basis"] == b) & (df[n_coeffs] == c), "RMSE"])
        y = np.reshape(y, (np.unique(x).shape[0], -1))
        y = np.mean(y, 1)
        x = np.unique(x)
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        x = -np.log10(x)
        # y = 1/1+np.exp(-)

        regr = linear_model.LogisticRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)

        print("{} & {} & {} & {} & {:.4f} \\\\".format(b, c, regr.coef_, regr.intercept_, r_square(y, y_pred)))
        # plt.figure()
        # plt.scatter(x, y)
        # plt.plot(x, m*x+i)

