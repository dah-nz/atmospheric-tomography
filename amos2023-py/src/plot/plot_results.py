import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from plot import data, colours

from tomography import psf

# df = data.load()

df_ra = data.load() # pd.concat([df_16_0, df_64_0])
df_ba = data.load() # pd.concat([df_16_0, df_64_0])
df_ia = data.load() # pd.concat([df_16_0, df_64_0])

# Plot settings
sns.set_theme(context="poster", style="whitegrid", font="serif", rc={"lines.linewidth": 1.25}) #, font_scale=2)
# sns.set_style("whitegrid")

label_coeffs = "N Coefficients"
label_series = "Asterism"

ast1_0 = 'Single star'
ast3_15 = '3 stars, 15" radius'
ast3_30 = '3 stars, 30" radius'
ast6_60 = '6 stars, 60" radius'
ast5_Asym = '5 stars asymmetric'
ideal = "Best estimate"
baseline = "Ground truth"

hue_order = [ast1_0, ast3_15, ast3_30, ast6_60, ast5_Asym, ideal, baseline]
col_order = ["Zernike", "DCT", "DST", "Wavelets (Haar)", "Wavelets (db2)"]

color_dict = colours.palette(
    [
        baseline,
        ast1_0,
        ast6_60,
        ast3_15,
        ast5_Asym,
        ideal,
        ast3_30,
    ]
)

df_ra.loc[df_ra["Asterism"] == "3_15", label_series] = ast3_15
df_ra.loc[df_ra["Asterism"] == "3_30", label_series] = ast3_30
df_ra.loc[df_ra["Asterism"] == "6_60", label_series] = ast6_60
df_ra.loc[df_ra["Asterism"] == "5_Asym", label_series] = ast5_Asym
df_ra.loc[df_ra["Asterism"] == "1_0", label_series] = ast1_0

lw = 2
df_ba = df_ba[df_ba["N_Coeffs"] == 16]

# Plot RMSE data
df_ra.loc[:, "RMSE"] = df_ra.loc[:, "RMSE_RA"]

df_ia.loc[:, "RMSE"] = df_ia.loc[:, "RMSE_IA"]
df_ia.loc[:, label_series] = ideal

df_ba.loc[:, "RMSE"] = df_ba.loc[:, "RMSE_BA"]
df_ba.loc[:, label_series] = baseline

df = pd.concat([df_ra, df_ia, df_ba])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="RMSE", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict,
    hue_order=hue_order, col_order=col_order,
    col_wrap=3
)

(g.set_axis_labels('Target Angle (")', 'RMSE (radians)')
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
    .set(ylim=(0.0, None), xlim=(0, None))
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))


# Plot FWHM data
telescope_diameter  = 0.6    # m
wavelength          = 600e-9 # m
f_number            = 6.5
um_per_m            = 1e6   # um per metre
resolution = psf.get_resolution(wavelength, telescope_diameter, f_number)#  * um_per_m

baseline = "Ground truth"
# df_ba.loc[:, "FWHM_Airy"] / 
df_ra.loc[:, "FWHM"] = df_ra.loc[:, "FWHM_RA"] * resolution

df_ia.loc[:, "FWHM"] = df_ia.loc[:, "FWHM_IA"] * resolution
df_ia.loc[:, label_series] = ideal

df_ba.loc[:, "FWHM"] = df_ba.loc[:, "FWHM_BA"] * resolution
df_ba.loc[:, label_series] = baseline

df = pd.concat([df_ra, df_ia, df_ba])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="FWHM", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict,
    hue_order=hue_order, col_order=col_order,
    col_wrap=3
)

(g.set_axis_labels('Target Angle (")', 'FWHM (")')
    .set(ylim=(None, None), xlim=(0, None))
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))


# Plot Strehl Data
strehl = lambda sigma: np.exp(-(sigma)**2)
df_ra.loc[:, "Strehl"] = strehl(df_ra.loc[:, "RMSE_RA"]) # df_ra.loc[:, "Strehl_RB"]

df_ia.loc[:, "Strehl"] = strehl(df_ia.loc[:, "RMSE_IA"]) # df_ia.loc[:, "Strehl_IB"]
df_ia.loc[:, label_series] = ideal

df_ba.loc[:, "Strehl"] = strehl(df_ba.loc[:, "RMSE_BA"]) # df_ba.loc[:, "Strehl_AB"]
df_ba.loc[:, label_series] = baseline

df = pd.concat([df_ra, df_ia, df_ba])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="Strehl", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict,
    hue_order=hue_order, col_order=col_order,
    col_wrap=3
)

(g.set_axis_labels('Target Angle (")', 'Strehl Ratio')
    .set(ylim=(None, None), xlim=(0, None))
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))

# Plot Strehl Data
df_ra.loc[:, "Strehl"] = df_ra.loc[:, "Strehl_RB"]

df_ia.loc[:, "Strehl"] = df_ia.loc[:, "Strehl_IB"]
df_ia.loc[:, label_series] = ideal

df_ba.loc[:, "Strehl"] = df_ba.loc[:, "Strehl_AB"]
df_ba.loc[:, label_series] = baseline

df = pd.concat([df_ra, df_ia, df_ba])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="Strehl", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict,
    hue_order=hue_order, col_order=col_order,
    col_wrap=3
)

(g.set_axis_labels('Target Angle (")', 'Strehl Ratio')
    .set(ylim=(None, None), xlim=(0, None))
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))


# Plot MSSIM data
df_ra.loc[:, "MSSIM"] = df_ra.loc[:, "MSSIM_RA"]

df_ia.loc[:, "MSSIM"] = df_ia.loc[:, "MSSIM_IA"]
df_ia.loc[:, label_series] = ideal

# df_ba.loc[:, "MSSIM"] = df_ba.loc[:, "MSSIM_AB"]
# df_ba.loc[:, label_series] = baseline

df = pd.concat([df_ra, df_ia])
df.rename(columns={"N_Coeffs": label_coeffs}, inplace=True)

g = sns.relplot(
    data=df, kind="line",
    x="Target_Gamma", y="MSSIM", col="Basis",
    hue=label_series, style=label_coeffs,
    errorbar=None, palette=color_dict,
    hue_order=hue_order[:-1], col_order=col_order,
    col_wrap=3
)

(g.set_axis_labels('Target Angle (")', 'MSSIM')
    .set(ylim=(None, None), xlim=(0, None))
    .set_titles("{col_name}")
    .tight_layout(w_pad=0.4)
)

for ax in g.axes.flat:
    ax.set_xticks(range(0, 91, 30))

# Plot violin plots of different basis functions
# df = pd.concat([df_16_0, df_64_0])
# df = df.loc[df["Target_Gamma"] == 0, :]

# g = sns.catplot(data=df, kind="violin", x="Asterism", y="RMSE_RA", hue="N_Coeffs", col="Basis", 
#     split=True, cut=0, col_wrap=3, facet_kws=dict(legend_out=False),)
# (g.set_axis_labels('Basis Function', 'RMSE (radians)')
#     .tight_layout()
# )
# sns.move_legend(g, "upper right", bbox_to_anchor=(1, 1))

# plt.figure()
# g = sns.violinplot(data=df, x="Basis", y="FWHM_RA", hue="Asterism", cut=0)

# plt.figure()
# g = sns.violinplot(data=df, x="Basis", y="Strehl_RB", hue="Asterism", cut=0)

# plt.figure()
# g = sns.violinplot(data=df, x="Basis", y="MSSIM_RB", hue="Asterism", cut=0)

plt.show()
