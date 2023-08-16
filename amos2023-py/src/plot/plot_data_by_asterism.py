import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

df_64_0 = pd.read_csv("./results/results_residuals_64_SNR_0.data")
df_16_0 = pd.read_csv("./results/results_residuals_16_SNR_0.data")
df_64_0.loc[:, "SNR"] = 0
df_16_0.loc[:, "SNR"] = 0

telescope_diameter =    0.6    # m
wavelength =            600e-9 # m
effective_focal_length = telescope_diameter * 6.5 # (f/6.5 system)
resolution = wavelength * effective_focal_length / 8

linewidth = 0.75

rmse_fig, rmse_axs = plt.subplots(1,5)
for i in range(5):
    rmse_axs[i].set_ylabel("RMSE (radians)")
    rmse_axs[i].set_xlabel('Angle (")')

fwhm_fig, fwhm_axs = plt.subplots(1,5)
for i in range(5):
    fwhm_axs[i].set_ylabel("FWHM (um)")
    fwhm_axs[i].set_xlabel('Angle (")')

strehl_fig, strehl_axs = plt.subplots(1,5)
for i in range(5):
    strehl_axs[i].set_ylabel("Strehl Ratio")
    strehl_axs[i].set_xlabel('Angle (")')

mssim_fig, mssim_axs = plt.subplots(1,5)
for i in range(5):
    mssim_axs[i].set_ylabel("MSSIM")
    mssim_axs[i].set_xlabel('Angle (")')

for n, df in [(64, df_64_0), (16, df_16_0)]:
    df.loc[:, "RMSE Diff"] = df.loc[:, "RMSE R:A"] - df.loc[:, "RMSE I:A"]
    df.loc[:, "Strehl"] = df.loc[:, "Max Int. R:A"] / df.loc[:, "Max Int. I:A"]
    df.loc[:, "FWHM Diff"] = df.loc[:, "FWHM R:A"] - df.loc[:, "FWHM I:A"]

    for i, basis in enumerate(df["Basis Function"].unique()):
        rmse_axs[i].set_title(basis)
        for asterism in df["N Guide Stars"].unique():
            rmse_i = []
            rmse_b = []
            rmse_r = []
            fwhm = []
            strehl = []
            mssim = []
            targets = sorted(list(df["Target Gamma"].unique()))
            for target in targets:
                rmse_i.append(df.loc[(df["Target Gamma"] == target) & (df["Basis Function"] == basis) & (df["N Guide Stars"] == asterism)]["RMSE I:A"].mean())
                rmse_b.append(df.loc[(df["Target Gamma"] == target) & (df["Basis Function"] == basis) & (df["N Guide Stars"] == asterism)]["RMSE B:A"].mean())
                rmse_r.append(df.loc[(df["Target Gamma"] == target) & (df["Basis Function"] == basis) & (df["N Guide Stars"] == asterism)]["RMSE R:A"].mean())

                fwhm.append(df.loc[(df["Target Gamma"] == target) & (df["Basis Function"] == basis) & (df["N Guide Stars"] == asterism)]["FWHM Diff"].mean())
                strehl.append(df.loc[(df["Target Gamma"] == target) & (df["Basis Function"] == basis) & (df["N Guide Stars"] == asterism)]["Strehl"].mean())
                mssim.append(df.loc[(df["Target Gamma"] == target) & (df["Basis Function"] == basis) & (df["N Guide Stars"] == asterism)]["MSSIM R:A-I:A"].mean())

            mssim_axs[i].plot(targets, mssim, label="{} Coeffs, {} Guidestars".format(n, asterism), linewidth=linewidth)
            strehl_axs[i].plot(targets, strehl, label="{} Coeffs, {} Guidestars".format(n, asterism), linewidth=linewidth)
            fwhm_axs[i].plot(targets, fwhm, label="{} Coeffs, {} Guidestars".format(n, asterism), linewidth=linewidth)
            rmse_axs[i].plot(targets, rmse_r, label="{} Coeffs, {} Guidestars".format(n, asterism), linewidth=linewidth)

        rmse_axs[i].plot(targets, rmse_i, '--', alpha = 0.2, label="{} Coeffs Ideal".format(n), linewidth=linewidth)
        if n == 16:
            rmse_axs[i].plot(targets, rmse_b, 'k', alpha = 0.2, label="Baseline", linewidth=linewidth)

rmse_axs[4].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
mssim_axs[4].legend()
strehl_axs[4].legend()
fwhm_axs[4].legend()

plt.show()