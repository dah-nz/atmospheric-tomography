import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from plot import data
from tomography import psf

df = data.load()

df0 = df.loc[df["Target_Gamma"] == 0, :].copy()

# Table for RMSE
print(" & & \multicolumn{2}{c}{Mean RMSE (radians)} \\\\")
print("Basis Function & Coefficients & Reconstruction & Ideal \\\\")
print("No estimate & - & {:.4f} & - \\\\".format(df0["RMSE_BA"].mean()))
for basis in df0["Basis"].unique():
    mean16 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["RMSE_RA", "RMSE_IA"]].mean()
    mean64 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["RMSE_RA", "RMSE_IA"]].mean()
    print("{} & {} & {:.4f} & {:.4f} \\\\".format(basis, 16, mean16["RMSE_RA"], mean16["RMSE_IA"]))
    print("& {} & {:.4f} & {:.4f} \\\\".format(64, mean64["RMSE_RA"], mean64["RMSE_IA"]))

resolution = psf.get_resolution(600e-9, 0.6, 6.5) * 1e6
print(" & & \multicolumn{2}{c}{Mean FWHM (um)} \\\\")
print("Basis Function & Coefficients & Reconstruction & Ideal \\\\")
print("Airy disk & - & {:.4f} & - \\\\".format(df0["FWHM_Airy"].mean() * resolution))
df0.loc[:, "FWHM_RRatio"] = df0.loc[:, "FWHM_Airy"] / df0.loc[:, "FWHM_RA"]
df0.loc[:, "FWHM_IRatio"] = df0.loc[:, "FWHM_Airy"] / df0.loc[:, "FWHM_IA"]
for basis in df0["Basis"].unique():
    mean16 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["FWHM_RRatio", "FWHM_IRatio"]].mean()
    mean64 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["FWHM_RRatio", "FWHM_IRatio"]].mean()
    print("{} & {} & {:.4f} & {:.4f} \\\\".format(basis, 16, mean16["FWHM_RRatio"], mean16["FWHM_IRatio"]))
    print("& {} & {:.4f} & {:.4f} \\\\".format(64, mean64["FWHM_RRatio"], mean64["FWHM_IRatio"]))

# for basis in df0["Basis"].unique():
#     t = stats.ttest_1samp(df0[df0["Basis"] == basis]['FWHM_IA'], popmean=df0["FWHM_Airy"].mean())
#     print(basis)
#     print(t.statistic)
#     print(t.pvalue)
#     print(t.df)
#     print(t.confidence_interval(0.95))

    # mean16 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["FWHM_RA", "FWHM_IA"]].mean()
    # mean64 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["FWHM_RA", "FWHM_IA"]].mean()
    # print("{} & {} & {:.4f} & {:.4f} \\\\".format(basis, 16, mean16["FWHM_RA"] * resolution, mean16["FWHM_IA"] * resolution))
    # print("& {} & {:.4f} & {:.4f} \\\\".format(64, mean64["FWHM_RA"] * resolution, mean64["FWHM_IA"] * resolution))

print(" & & \multicolumn{2}{c}{Mean Strehl Ratio} \\\\")
print("Basis Function & Coefficients & Reconstruction & Ideal \\\\")
for basis in df0["Basis"].unique():
    mean16 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["Strehl_RB", "Strehl_IB"]].mean()
    mean64 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["Strehl_RB", "Strehl_IB"]].mean()
    print("{} & {} & {:.4f} & {:.4f} \\\\".format(basis, 16, mean16["Strehl_RB"], mean16["Strehl_IB"]))
    print("& {} & {:.4f} & {:.4f} \\\\".format(64, mean64["Strehl_RB"], mean64["Strehl_IB"]))

print(" & & \multicolumn{2}{c}{Mean MSSIM} \\\\")
print("Basis Function & Coefficients & Reconstruction & Ideal \\\\")
for basis in df0["Basis"].unique():
    mean16 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["MSSIM_RB", "MSSIM_IB"]].mean()
    mean64 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["MSSIM_RB", "MSSIM_IB"]].mean()
    print("{} & {} & {:.4f} & {:.4f} \\\\".format(basis, 16, mean16["MSSIM_RB"], mean16["MSSIM_IB"]))
    print("& {} & {:.4f} & {:.4f} \\\\".format(64, mean64["MSSIM_RB"], mean64["MSSIM_IB"]))