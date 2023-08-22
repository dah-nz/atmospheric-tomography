import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import data, colours
from tomography import psf

df = data.load_noise()

sns.set_theme(context="poster", style="whitegrid", font="serif")

clr_dict = colours.palette()

r = psf.get_resolution(600e-9, 0.6, 6.5) * 1e6

for snr in df["SNR"].unique():
    if snr != 0:
        df.loc[df["SNR"] == snr, "RMSE"] = df.loc[df["SNR"] == snr, "RMSE_RA"] # / df.loc[df["SNR"] == 0, "RMSE_RA"]
        df.loc[df["SNR"] == snr, "FWHM"] = df.loc[df["SNR"] == snr, "FWHM_RA"] * r # / df.loc[df["SNR"] == 0, "FWHM_RA"]
        df.loc[df["SNR"] == snr, "MSSIM"] = df.loc[df["SNR"] == snr, "MSSIM_RB"] # / df.loc[df["SNR"] == 0, "MSSIM_RB"]
        df.loc[df["SNR"] == snr, "Strehl"] = df.loc[df["SNR"] == snr, "Strehl_RB"] # / df.loc[df["SNR"] == 0, "Strehl_RB"]

df = df.loc[df["SNR"] != 0, :]

df = df.loc[(df["Target_Gamma"] == 0) | (df["Target_Gamma"] == 30) | (df["Target_Gamma"] == 75), :]

n_coeffs = "N Coefficients"
gamma = "Target Angle (\")"
df.rename(columns={"N_Coeffs": n_coeffs}, inplace=True)
df.rename(columns={"Target_Gamma": gamma}, inplace=True)

# Plot data
for y_data, y_units in [("FWHM", "FWHM (um)"), ("RMSE", "RMSE (radians)"), ("MSSIM", "MSSIM"), ("Strehl", "Strehl Ratio")]:
    g = sns.relplot(
        data=df, kind='line',
        y=y_data, x="SNR", col="Basis",
        style=n_coeffs, hue=gamma,
        errorbar=None, palette=clr_dict
    )

    (
        g.set_axis_labels('SNR (dB)', y_units)
        .set_titles("{col_name}")
        .tight_layout(w_pad=0.4)
    )

    for ax in g.axes.flat:
        ax.invert_xaxis()

plt.show()