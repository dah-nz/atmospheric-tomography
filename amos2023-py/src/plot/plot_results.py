import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from plot import data, colours

from tomography import psf

df = data.load()

df_ra = data.load() # pd.concat([df_16_0, df_64_0])
df_ba = data.load() # pd.concat([df_16_0, df_64_0])
df_ia = data.load() # pd.concat([df_16_0, df_64_0])

# Plot settings
sns.set_theme(context="poster", style="whitegrid", font="serif") #, font_scale=2, rc={"lines.linewidth": 2.5})
# sns.set_style("whitegrid")

label_coeffs = "N Coefficients"
label_series = "Asterism"

ast3 = '3 stars at 30" radius'
ast6 = '6 stars at 60" radius'
ast5 = '5 stars asymmetric'
ideal = "Best estimate"
baseline = "No estimate"
airydisk = "No distortion (Airy disk)"

color_dict = colours.palette(
    [
        ast3,
        ast6,
        ast5,
        ideal,
        baseline,
        airydisk
    ]
)

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
