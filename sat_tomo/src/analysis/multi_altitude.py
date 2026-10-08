import pandas as pd
import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def consolidate_data():
    # directories = glob.glob("D:/Data/2026/altitudes3")
    dir = "D:/Data/2026/altitudes3"
    df = pd.DataFrame()

    # for dir in directories:
    results_file = os.path.join(dir, "current/results.csv")
    df0 = pd.read_csv(results_file)
    df0 = df0.rename(columns={"Epoch": "AltitudeIndex"})
    df0 = df0[df0["Type"] == "Reconstruction"]
    df0 = df0.drop(columns=["TrainingSet", "Screens", "Stage", "Unnamed: 0", "Asterism", "Type"])
    print(df0.head())

    phase_file = os.path.join(dir, "current/phase.csv")
    df1 = pd.read_csv(phase_file)
    df1 = df1[df1["Layer"] != "Total"]
    df1 = df1.drop(columns=["Epoch", "Unnamed: 0", "Variance", "r0", "Index"]).drop_duplicates().reset_index()
    df1 = df1.pivot(index="AltitudeIndex", columns="Layer", values="Altitude").reset_index()
    df1 = df1.rename(columns={"1": "Lower", "2": "Middle", "3": "Upper"})
    print(df1.head())

    df = df0.merge(df1, on="AltitudeIndex", how="left")
    df = df.drop(columns=["AltitudeIndex"])
    df["Lower"] = pd.to_numeric(df["Lower"]).astype(int)
    df["Middle"] = pd.to_numeric(df["Middle"]).astype(int)
    df["Upper"] = pd.to_numeric(df["Upper"]).astype(int)

    # print(df.head())

    df = df.rename(columns={"Loss": "MSE"})
    df = df.groupby(["Model", "Lower", "Middle", "Upper"])["MSE"].median().reset_index()
    df["Model"] = df["Model"].map({
        'AO': "SCAO",
        'GuideStarEmbed_NoInverseKzz_Variable_Final': "MINN-PQ",
        'GuideStarEmbed_baseline_Variable_Final': "Direct MLNN",
        'GuideStarEmbed_pseudo_lsq_Variable_Final': "MINN-LS",
        'GuideStarEmbed_pseudo_mmse_Variable_Final': "MINN-MMSE",
        'ResNetEmbed_Variable_Final': "ResNet",
        'LeastSquares': "Least Squares"
    }).fillna(df["Model"])

    keys = ["Lower", "Upper", "Middle"]

    reference = (
        df[df["Model"] == "Fitting Error"]
        .set_index(keys)["MSE"]
    )

    df["Ratio"] = (
        df["MSE"] /
        df.set_index(keys).index.map(reference)
    )

    print(df["Model"].unique())
    print(df.head())

    save_dir = "//file/UsersD$/dah158/Home/My Documents/03 - Doctorate/2026/Journal Paper/fig"
    sns.set_theme(context='paper', style='whitegrid', font='serif')
    fig_height = 2.4
    fig_width = 2.4
    fig_aspect = fig_width/fig_height

    ylim_loss = (10, 50)
    ylim_ratio = (0, 10)

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111)

    # df = df[df["Model"].isin(["MINN-LS", "MMSE", "Least Squares", "MINN-PQ", "Direct MLNN", "Open Loop", "Fitting Error"])]
    df["l1_scaled"] = df["Lower"] / 1000
    df["l2_scaled"] = df["Middle"] / 1000
    df["l3_scaled"] = df["Upper"] / 1000
    # df = df[df["Model"].isin(["MINN-LS", "SCAO", "GLAO", "Fitting Error"])]

    df0 = df[df["Upper"] == 10000]
    df0 = df0[df0["Middle"] == 5000]

    # hue_order = ["Open Loop", "MMSE", "Least Squares", "SCAO", "GLAO", "Direct MLNN", "ResNet", "MINN-LS", "MINN-PQ", "Fitting Error"]

    hue_order = ["MMSE", "Direct MLNN", "MINN-LS", "MINN-PQ"]

    g = sns.relplot(data=df0, kind='line',
                    x='l1_scaled', y='MSE', hue="Model", hue_order=hue_order,
                    height=fig_height, aspect=fig_aspect, legend=False,
                    facet_kws={'sharey': "row", 'sharex': True})
    g.set(ylabel="Loss (rads$^2$)", xlabel="Layer 1 Altitude (km)", xlim=(0.00, 3.00), ylim=ylim_loss)
    plt.savefig(os.path.join(save_dir, f"altitude_loss_lower.pgf"), backend='pgf')

    g = sns.relplot(data=df0, kind='line',
                    x='l1_scaled', y='Ratio', hue="Model", hue_order=hue_order,
                    height=fig_height, aspect=fig_aspect,
                    facet_kws={'sharey': "row", 'sharex': True})
    g.set(ylabel="Loss Ratio", xlabel="Layer 1 Altitude (km)", ylim=ylim_ratio)
    plt.savefig(os.path.join(save_dir, f"altitude_ratio_lower.pgf"), backend='pgf')

    df0 = df[df["Lower"] == 1000]
    df0 = df0[df0["Middle"] == 5000]
    g = sns.relplot(data=df0, kind='line',
                    x='l3_scaled', y='MSE', hue="Model", hue_order=hue_order,
                    height=fig_height, aspect=2.6/fig_height,
                    facet_kws={'sharey': "row", 'sharex': True})
    g.set(ylabel="Loss (rads$^2$)", xlabel="Layer 3 Altitude (km)", xlim=(5.25, 15.00), ylim=ylim_loss)
    plt.savefig(os.path.join(save_dir, f"altitude_loss_upper.pgf"), backend='pgf')

    g = sns.relplot(data=df0, kind='line',
                    x='l3_scaled', y='Ratio', hue="Model", hue_order=hue_order,
                    height=fig_height, aspect=fig_aspect,
                    facet_kws={'sharey': "row", 'sharex': True})
    g.set(ylabel="Loss Ratio", xlabel="Layer 3 Altitude (km)", ylim=ylim_ratio)
    plt.savefig(os.path.join(save_dir, f"altitude_ratio_upper.pgf"), backend='pgf')

    df0 = df[df["Lower"] == 1000]
    df0 = df0[df0["Upper"] == 10000]
    g = sns.relplot(data=df0, kind='line',
                    x='l2_scaled', y='MSE', hue="Model", hue_order=hue_order,
                    height=fig_height, aspect=fig_aspect, legend=False,
                    facet_kws={'sharey': "row", 'sharex': True})
    g.set(ylabel="Loss (rads$^2$)", xlabel="Layer 2 Altitude km)", xlim=(1.25, 9.75), ylim=ylim_loss)
    plt.savefig(os.path.join(save_dir, f"altitude_loss_middle.pgf"), backend='pgf')

    g = sns.relplot(data=df0, kind='line',
                    x='l2_scaled', y='Ratio', hue="Model", hue_order=hue_order,
                    height=fig_height, aspect=fig_aspect,
                    facet_kws={'sharey': "row", 'sharex': True})
    g.set(ylabel="Loss Ratio", xlabel="Layer 2 Altitude (km)", ylim=ylim_ratio)
    plt.savefig(os.path.join(save_dir, f"altitude_ratio_middle.pgf"), backend='pgf')


    # df = df[df["Model"] == "MINN-LS"]

    # Apply seaborn aesthetics
    # for lower in [0, 1]:
    #     df0 = df[df["Lower"] == lower]
    #     # Initialize 3D plot
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection="3d")

    #     # Map seaborn palette to categories
    #     palette = sns.color_palette("Set2")
    #     for i, model in enumerate(df0["Model"].unique()):
    #     # colors = df0["Model"].map({"SCAO": palette[0],
    #     #                         "GLAO": palette[1],
    #     #                         "Fitting Error": palette[2],
    #     #                         "Direct MLNN": palette[3],
    #     #                         "MINN-LS": palette[4],
    #     #                         "MINN-MMSE": palette[5],
    #     #                         "MINN-PQ": palette[6]})

    #     # Plot data
    #         df1 = df0[df0["Model"] == model]
    #         X = df1["Middle"].unique()
    #         Y = df1["Upper"].unique()

    #         X, Y = np.meshgrid(X, Y)

    #         Z = df1["MSE"].values.reshape(len(Y), len(X))

    #     # ax.scatter(df0["Middle"], df0["Upper"], df0["MSE"], c=colors, s=50)
    #         ax.plot_surface(X, Y, Z, color=palette[i], label=model)


    #     # Set labels
    #     ax.set_xlabel("Middle Layer (km)")
    #     ax.set_ylabel("Upper Layer (km)")
    #     ax.set_zlabel("MSE (rads$^2$)")
    #     ax.legend()

    plt.show()


def display():
    consolidate_data()

if __name__ == "__main__":
    display()