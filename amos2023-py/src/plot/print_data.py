import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

df_64_0 = pd.read_csv("./results/results_residuals_64_SNR_0.data")
df_16_0 = pd.read_csv("./results/results_residuals_16_SNR_0.data")
df_64_0.loc[:, "SNR"] = 0
df_16_0.loc[:, "SNR"] = 0

f = open("data.csv", 'w')

for t, df in [("Coeffs: 64; SNR: 0", df_64_0), ("Coeffs: 16; SNR: 0", df_16_0)]:
    df.loc[:, "Strehl"] = df.loc[:, "Max Int. R:A"] / df.loc[:, "Max Int. I:A"]
    df.loc[:, "RMSE Baseline"] = df.loc[:, "RMSE B:A"] - df.loc[:, "RMSE R:A"]
    df.loc[:, "RMSE Diff"] = df.loc[:, "RMSE R:A"] - df.loc[:, "RMSE I:A"]
    df.loc[:, "FWHM Diff"] = df.loc[:, "FWHM R:A"] - df.loc[:, "FWHM I:A"]
    f.write(t + "\n")
    for basis in df["Basis Function"].unique():
        f.write("Basis: {}\n".format(basis))
        for target in df["Target Gamma"].unique():
            f.write("Target: {}\n".format(target))
            out_df = df.loc[(df["Target Gamma"] == target) & (df["Basis Function"] == basis)][["RMSE R:A", "RMSE B:A", "RMSE I:A", "FWHM R:A", "FWHM I:A", "FWHM Diff", "Strehl", "MSSIM R:A-I:A"]].agg(["mean", "std"])
            test_df = df.loc[(df["Target Gamma"] == target) & (df["Basis Function"] == basis)]
            d = np.array(test_df.loc[:, "RMSE R:A"])
            ci = stats.t.interval(0.95, d.shape[0], loc=np.mean(d), scale=stats.sem(d))
            out_df["95% CI LB"] = ci[0]
            out_df["95% CI UB"] = ci[1]
            f.write(out_df.to_string() + "\n")
            # f.write("95% Conf Interval :{}\n".format(ci))
            # fitting = stats.ttest_rel(test_df.loc[:, "RMSE R:A"], test_df.loc[:, "RMSE B:A"], alternative="less")
            # f.write("Fitting Error T-Test pValue, {:1.2e}\n".format(fitting.pvalue))
            # tomo = stats.ttest_rel(test_df.loc[:, "RMSE R:A"], test_df.loc[:, "RMSE I:A"], alternative="greater")
            # f.write("Tomography Error T-Test pValue, {:1.2e}\n".format(tomo.pvalue))
            # fwhm = stats.ttest_rel(test_df.loc[:, "FWHM R:A"], test_df.loc[:, "FWHM I:A"], alternative="greater")
            # f.write("FWHM Error T-Test pValue, {:1.2e}\n".format(fwhm.pvalue))

# Compare 
# df = pd.concat([df_64_0, df_16_0])
# coeffs = df["N Coeffs"].unique()
# bf = df["Basis Function"].unique()
# sets = []
# for n in coeffs:
#     for b in bf:
#         sets.append((b,n))

# print(sets)

# for ab, an in sets:
#     for bb, bn in sets:
#         if an is not bn or ab is not bb:
#             test_a = df.loc[(df["Target Gamma"] == 0) & (df["N Coeffs"] == an) & (df["Basis Function"] == ab)]
#             test_b = df.loc[(df["Target Gamma"] == 0) & (df["N Coeffs"] == bn) & (df["Basis Function"] == bb)]
#             test = stats.ttest_rel(test_a.loc[:, "RMSE R:A"], test_b.loc[:, "RMSE R:A"])
#             conf = stats.t.interval()
#             f.write("{}:{} and {}:{}\n".format(ab, an, bb, bn))
#             f.write("T-Test, {}\n".format(test.pvalue))
            

f.close()