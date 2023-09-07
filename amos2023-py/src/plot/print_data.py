import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from plot import data
from tomography import psf

df = data.load()

df0 = df.loc[(df["Target_Gamma"] == 0) & (df["Asterism"] == "3_15"), :].copy()

# Table for RMSE
m = df0["RMSE_BA"].mean()
print(" & & \multicolumn{2}{c}{ RMSE (radians) }&  \\\\")
print("Basis Function & Modes & Best & Reconstruction & Improvement (\\%) \\\\")
print("\\hline")
print("No estimate    &   -   &  -   & {:.4f}         &    -            \\\\".format(m))
for basis in df0["Basis"].unique():
    mean16 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 16)][["RMSE_RA", "RMSE_IA"]].mean()
    mean64 = df0.loc[(df0["Basis"] == basis) & (df0["N_Coeffs"] == 64)][["RMSE_RA", "RMSE_IA"]].mean()
    print("{} & {} & {:.4f} & {:.4f} & {:.1f} \\\\".format(basis,  16, mean16["RMSE_IA"], mean16["RMSE_RA"], 100*(m-mean16["RMSE_RA"])/m))
    print("   & {} & {:.4f} & {:.4f} & {:.1f} \\\\".format(        64, mean64["RMSE_IA"], mean64["RMSE_RA"], 100*(m-mean64["RMSE_RA"])/m))