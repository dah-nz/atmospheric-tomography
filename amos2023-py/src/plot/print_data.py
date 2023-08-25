import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from plot import data
from tomography import psf

df = data.load()

df0 = df.loc[(df["Target_Gamma"] == 0) & (df["Asterism"] != "1_0"), :].copy()

# Table for RMSE
print(" & & \multicolumn{2}{c}{RMSE (radians)}& \multicolumn{2}{c}{MSSIM} \\\\")
print("Basis Function & Coefficients & Reconstruction & Ideal & Reconstruction & Ideal \\\\")
print("No estimate & - & {:.4f} & - & {:.4f} & - \\\\".format(df0["RMSE_BA"].mean(),df0["MSSIM_BA"].mean()))
for basis in df0["Basis"].unique():
    mean16 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["RMSE_RA", "RMSE_IA", "MSSIM_RA", "MSSIM_IA"]].mean()
    mean64 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["RMSE_RA", "RMSE_IA", "MSSIM_RA", "MSSIM_IA"]].mean()
    print("{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(basis, 16, mean16["RMSE_RA"], mean16["RMSE_IA"], mean16["MSSIM_RA"], mean16["MSSIM_IA"]))
    print("& {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(64, mean64["RMSE_RA"], mean64["RMSE_IA"], mean64["MSSIM_RA"], mean64["MSSIM_IA"]))

resolution = psf.get_resolution(600e-9, 0.6, 6.5)
print(" & & \multicolumn{2}{c|}{FWHM (\")} & \multicolumn{2}{c}{Strehl Ratio}\\\\")
print("Basis Function & Coefficients & Reconstruction & Ideal & Reconstruction & Ideal \\\\")
print("Airy disk & | & | & {:.4f} & | & | \\\\".format(df0["FWHM_Airy"].mean() * resolution))
print("No estimate & | & {:.4f} & | & {:.4f} & | \\\\".format(df0["FWHM_BA"].mean() * resolution, df0["Strehl_AB"].mean()))
for basis in df0["Basis"].unique():
    mean16 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["FWHM_RA", "FWHM_IA"]].mean() * resolution
    mean64 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["FWHM_RA", "FWHM_IA"]].mean() * resolution
    mean16s = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["Strehl_RB", "Strehl_IB"]].mean()
    mean64s = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["Strehl_RB", "Strehl_IB"]].mean()
    print("{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(basis, 16, mean16["FWHM_RA"], mean16["FWHM_IA"], mean16s["Strehl_RB"], mean16s["Strehl_IB"]))
    print("& {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(64, mean64["FWHM_RA"], mean64["FWHM_IA"], mean64s["Strehl_RB"], mean64s["Strehl_IB"]))

