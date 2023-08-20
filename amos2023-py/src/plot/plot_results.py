import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tomography import psf

# Plot data for SNR = 0, different asterisms

df_16_0 =       pd.read_csv("./results/results_home_16_SNR_0.data")
df_64_0_z =     pd.read_csv("./results/results_home_64_SNR_0_z.data")
df_64_0_dct =   pd.read_csv("./results/results_home_64_SNR_0_dct.data")
df_64_0_dst =   pd.read_csv("./results/results_home_64_SNR_0_dst.data")
df_64_0_wd =    pd.read_csv("./results/results_home_64_SNR_0_wd.data")
df_64_0_wh =    pd.read_csv("./results/results_home_64_SNR_0_wh.data")

df_64_0 = pd.concat([df_64_0_z, df_64_0_dct, df_64_0_dst, df_64_0_wd, df_64_0_wh])

df_ra = pd.concat([df_16_0, df_64_0])
df_ba = pd.concat([df_16_0, df_64_0])
df_ia = pd.concat([df_16_0, df_64_0])

# Plot settings
sns.set_context("poster") #, font_scale=2, rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")

label_coeffs = "N Coefficients"
label_series = "Asterism"

ast3 = '3 stars at 30" radius'
ast6 = '6 stars at 60" radius'
ast5 = '5 stars asymmetric'
ideal = "Best estimate"
baseline = "No estimate"

c0 = "#2D4059FF"
c1 = "#EA5455FF"
c2 = "#EFCB68FF"
c3 = "#00A9A580"
c4 = "#95B2B880"

lw = 2

# Plot RMSE data
df_ra.loc[:, "RMSE"] = df_ra.loc[:, "RMSE_RA"]
df_ra.loc[df_ra["Asterism"] == 3, label_series] = ast3
df_ra.loc[df_ra["Asterism"] == 6, label_series] = ast6
df_ra.loc[df_ra["Asterism"] == 5, label_series] = ast5

df_ia.loc[:, "RMSE"] = df_ia.loc[:, "RMSE_IA"]
df_ia.loc[:, label_series] = ideal

df_ba = df_ba[df_ba["N_Coeffs"] == 16]
df_ba.loc[:, "RMSE"] = df_ba.loc[:, "RMSE_BA"]
df_ba.loc[:, label_series] = baseline

df = pd.concat([df_ra, df_ia, df_ba])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

color_dict = { ast3: c0, ast6: c1, ast5: c2, ideal: c3, baseline: c4 }

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="RMSE", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict
)

(g.set_axis_labels('Target Angle (")', 'RMSE (radians)')
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))


# Plot FWHM data
telescope_diameter  = 0.6    # m
wavelength          = 600e-9 # m
f_number            = 6.5
um_per_m            = 1e6   # um per metre
resolution = psf.get_resolution(wavelength, telescope_diameter, f_number) * um_per_m

baseline = "No distortion (Airy disk)"
color_dict = { ast3: c0, ast6: c1, ast5: c2, ideal: c3 }

df_ra.loc[:, "FWHM"] = df_ra.loc[:, "FWHM_Airy"] / df_ra.loc[:, "FWHM_RA"] # * resolution
df_ra.loc[df_ra["Asterism"] == 3, label_series] = ast3
df_ra.loc[df_ra["Asterism"] == 6, label_series] = ast6
df_ra.loc[df_ra["Asterism"] == 5, label_series] = ast5

df_ia.loc[:, "FWHM"] = df_ia.loc[:, "FWHM_Airy"] / df_ia.loc[:, "FWHM_IA"] # * resolution
df_ia.loc[:, label_series] = ideal

# df_ba = df_ba[df_ba["N_Coeffs"] == 16]
# df_ba.loc[:, "FWHM"] = df_ba.loc[:, "FWHM_Airy"] * resolution
# df_ba.loc[:, label_series] = baseline

df = pd.concat([df_ra, df_ia])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="FWHM", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict
)

(g.set_axis_labels('Target Angle (")', 'FWHM (ratio)')
    .set(ylim=(None, 1.0))
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))


# Plot Strehl Data
color_dict = { ast3: c0, ast6: c1, ast5: c2, ideal: c3, baseline: c4 }
df_ra.loc[:, "Strehl"] = df_ra.loc[:, "Strehl_RB"]
df_ra.loc[df_ra["Asterism"] == 3, label_series] = ast3
df_ra.loc[df_ra["Asterism"] == 6, label_series] = ast6
df_ra.loc[df_ra["Asterism"] == 5, label_series] = ast5

df_ia.loc[:, "Strehl"] = df_ia.loc[:, "Strehl_IB"]
df_ia.loc[:, label_series] = ideal

df = pd.concat([df_ra, df_ia])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="Strehl", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict
)

(g.set_axis_labels('Target Angle (")', 'Strehl Ratio')
    .set(ylim=(None, 1))
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))


# Plot MSSIM data
baseline = "No distortion (Airy disk)"
color_dict = { ast3: c0, ast6: c1, ast5: c2, ideal: c3, baseline: c4 }

df_ra.loc[:, "MSSIM"] = df_ra.loc[:, "MSSIM_RB"]
df_ra.loc[df_ra["Asterism"] == 3, label_series] = ast3
df_ra.loc[df_ra["Asterism"] == 6, label_series] = ast6
df_ra.loc[df_ra["Asterism"] == 5, label_series] = ast5

df_ia.loc[:, "MSSIM"] = df_ia.loc[:, "MSSIM_IB"]
df_ia.loc[:, label_series] = ideal

df = pd.concat([df_ra, df_ia])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="MSSIM", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict
)

(g.set_axis_labels('Target Angle (")', 'MSSIM')
    .set(ylim=(None, 1.0))
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))

plt.show()
quit()
# Plot data for SNR changing, aggregate asterisms

df_16_0 = pd.read_csv("./results/results_mean_shifted_16_SNR_0.data")
df_16_3 = pd.read_csv("./results/results_mean_shifted_16_SNR_3.data")
df_16_10 = pd.read_csv("./results/results_mean_shifted_16_SNR_10.data")
df_16_20 = pd.read_csv("./results/results_mean_shifted_16_SNR_20.data")
df_16_30 = pd.read_csv("./results/results_mean_shifted_16_SNR_30.data")
# df_64_0 = pd.read_csv("./results/results_mean_shifted_64_SNR_0.data")
# df_64_3 = pd.read_csv("./results/results_mean_shifted_64_SNR_3.data")

baseline = "No estimate"
label_series = "Basis"

# Plot RMSE data
DFs = [df_16_3, df_16_10, df_16_20, df_16_30]
for df in DFs:
    df["RMSE %"] = df["RMSE_RA"] - df_16_0["RMSE_RA"]

df = pd.concat(DFs)
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

color_dict = { "Zernike": c0, "DCT": c1, "DST": c2, "Wavelets (Haar)": c3, "Wavelets (db2)": c4, baseline: "#00000080" }

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="RMSE %", col="SNR",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict
)

(g.set_axis_labels('Target Angle (")', 'RMSE (radians)')
    .set_titles("SNR: {col_name}dB")
    .tight_layout(w_pad=0.4)
)

plt.show()

