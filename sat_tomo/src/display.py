import matplotlib.pyplot as plt
import astronomy

import numpy as np
import pandas as pd
import seaborn as sns

import os

def asterisms():
    magnitudes = range(6, 14 + 1)
    plt.figure()

    ave_arcs = []
    ave_mags = []

    for magnitude in magnitudes:
        angles = astronomy.satellite.load_map(magnitude)

        pairs = []

        for angle in angles:
            for a, _, m in angle:
                pairs.append((a, m))

        indices = np.random.choice(range(len(pairs)), 1000)
        mags = [pairs[i][1] for i in indices]
        arcs = [pairs[i][0] for i in indices]

        ave_arcs.append(np.mean(arcs))
        ave_mags.append(np.mean(mags))

        plt.scatter(mags, arcs, marker='+', alpha=1.0/(len(magnitudes)-1))
        plt.plot(np.mean(mags), np.mean(arcs), marker='o', label=f"M{magnitude}")

    plt.plot(ave_mags, ave_arcs, 'k--', alpha=0.5)
    plt.yscale("log")
    plt.legend()

def training(cfg, file_name="current/results.csv"):
    with open(os.path.join(cfg.directory, file_name), "r") as f:
        df = pd.read_csv(f)

    df_v = df.drop(df[df.Stage != "Verification"].index)
    df = df.drop(df[df.Model == "Open Loop"].index)
    df = df.drop(df[df.Stage != "Train"].index)

    df = df.sort_values(by=["Epoch"])

    ma_windows = [16]

    for window_size in ma_windows:

        df[f'ma_{window_size}'] = (
            df.groupby(['Type', 'Model', 'Stage', 'Asterism'])['Loss']
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        )

    df_long = df.melt(
        id_vars=["Epoch", "Stage", "Model", "Screens", "Asterism", "Type", "TrainingSet"],
        value_vars=[f"ma_{w}" for w in ma_windows],
        var_name="Window Size",
        value_name="Moving Average"
    )

    df_long["Window Size"] = df_long["Window Size"].str.extract('(\d+)').astype(int)
    df_long = df_long[df_long['Window Size'] == ma_windows[0]]
    df_gs = df_long[df_long['Type'] == "Guidestar Loss"]
    df_long = df_long[df_long["Type"] == "Reconstruction"]

    # print(df[(df["Type"]=="Coefficients") & (df["Stage"]=="Train")&(df['Model']=='MLNN_baseline')].tail(12))
    if not df_gs.empty:
        print(df_gs.head())
        g = sns.relplot(data=df_gs, kind='line', 
                    x='Epoch', y='Moving Average', row="Type",
                    style="Asterism", hue="Model",
                    facet_kws={'sharey': "row", 'sharex': True})
        
        g.figure.suptitle(f"Guidestar Reconstruction Loss")

    print(df_long.head())

    for comparison in ['GuideStarEmbed', 'MLNN']:
        df = df_long[~df_long['Model'].str.contains(comparison)]
        df = df[df['Asterism'] == "Variable"]

        g = sns.relplot(data=df, kind='line', 
                    x='Epoch', y='Moving Average', row="Type",
                    style="Asterism", hue="Model",
                    facet_kws={'sharey': "row", 'sharex': True})
        g.figure.subplots_adjust(top=0.9)
    
    # Remove all but last verification results
    try:
        max_training_set = df_v["TrainingSet"].max()
        df_v = df_v.drop(df_v[df_v.TrainingSet != max_training_set].index)
        df_v = df_v[df_v["Asterism"] == "Variable"]
    except KeyError:
        pass


    
    df_v_recon = df_v.drop(df_v[df_v.Type != "Reconstruction"].index)

    for comparison in ['GuideStarEmbed', 'MLNN']:
        fig, axs = plt.subplots(1, 1)
        df = df_v_recon[~df_v_recon['Model'].str.contains(comparison)]
        df = df[~df['Model'].str.contains("_e")]
        sns.boxplot(data=df, x='Model', y='Loss', ax=axs)
        axs.set_title("Reconstruction")
        axs.tick_params(axis='x', rotation=30)
    
    # df_v_coeff = df_v.drop(df_v[df_v.Type != "Coefficients"].index)
    # sns.boxplot(data=df_v_coeff, x='Model', y='Loss', ax=axs[1])
    # axs[1].set_title("Coefficients")
    # axs[1].tick_params(axis='x', rotation=30)

    for comparison in ['GuideStarEmbed', 'MLNN', 'Variable']:
        if comparison == "Variable":
            df = df_v_recon[~df_v_recon['Model'].str.contains(comparison)]
        else:
            df = df_v_recon[df_v_recon['Model'].str.contains(comparison)]

        print("*** Reconstruction Statistics ***")
        print(df[["Model", "Loss"]].groupby("Model").describe())

    # print("*** Coefficient Statistics ***")
    # print(df_v_coeff[["Model", "Loss"]].groupby("Model").describe())
    
def main():
    file_name = "current/results.csv"
    r0_type = ["different"]
    Dr0_values = [10, 20, 30, 40]

    dfs = []

    for r0 in r0_type:
        for Dr0 in Dr0_values:
            directory = f"D:/Data/2026/{r0}_r0_Dr0{Dr0}"

            with open(os.path.join(directory, file_name), "r") as f:
                df0 = pd.read_csv(f)
                df0["r0_Type"] = r0
                df0["D_r0"] = Dr0
                dfs.append(df0)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by=["Epoch"])

    window_size = 10

    df[f'Moving Average'] = (
        df.groupby(['Type', 'Model', 'Stage', 'Asterism', 'r0_Type', 'D_r0'])['Loss']
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    )

    print(df.columns)

    print(df['Stage'].unique())
    print(df['Model'].unique())
    print(df['Screens'].unique())
    print(df['Asterism'].unique())
    print(df['Type'].unique())
    print(df['TrainingSet'].unique())
    print(df['r0_Type'].unique())
    print(df['D_r0'].unique())




    df["Embed"] = "No"
    df.loc[df['Model'].str.contains("Embed"), "Embed"] = "Yes"
    df.loc[df['Model'].isin(['Open Loop', 'LeastSquares', 'MMSE', "Fitting Error", 'GLAO', 'AO']), "Embed"] = "NA"
    print(df['Embed'].unique())

    df["Size"] = "Standardised"
    df.loc[df['Model'].str.contains("smol"), "Size"] = "Small"
    print(df['Size'].unique())


    df["Model"] = df["Model"].replace(r'(?:^GuideStarEmbed|Embed)_?', '', regex=True)
    df["Model"] = df["Model"].replace(r'smol_', '', regex=True)
    df["Model"] = df["Model"].replace(r'MLNN_', '', regex=True)
    df["Model"] = df["Model"].replace(r'ResNet(?!_|$)', 'ResNet_', regex=True)
    df["Save_Epoch"] = df["Model"].str.extract(r'_e(\d+)').fillna("Final")
    #  = np.where(extracted.notna(), extracted, "Final")
    print(df["Save_Epoch"].unique())
    df["Model"] = df["Model"].replace(r'_Variable.*', '', regex=True)
    df["Model"] = df["Model"].replace(r'_smol', '', regex=True)

    df = df[df['Model'] != 'singlemode']

    df["Model"] = df["Model"].map({
        'baseline': "Direct MLNN",
        'pseudo_lsq': "Pseudo-LS",
        'pseudo_mmse': "Pseudo-MMSE",
        'NoInverseKzz': "PQ + Priors",
        'LeastSquares': "Least Squares"
    }).fillna(df["Model"])

    print(df['Model'].unique())

    # df['Model'] = pd.Categorical(df['Model'], categories=df['Model'].unique())
    # df['Model'] = df['Model'].cat.reorder_categories(['Fitting Error', 
    #                 'AO',
    #                 'GLAO', 
    #                 'Least Squares',
    #                 'MMSE',
    #                 'Baseline',
    #                 'Pseudo Least Square',
    #                 'Pseudo MMSE',
    #                 'Pseudo MMSE (No Inverse)',
    #                 'ResNet',
    #                 'Single Mode',
    #                 'Open Loop'],
    #                 ordered=True)
    
    for D_r0 in df["D_r0"].unique():
        for r0_Type in df["r0_Type"].unique():
            df0 = df[(df["Save_Epoch"] == "Final") 
                    & (df["Size"] == "Standardised") 
                    & (df["Type"] == "Reconstruction")
                    & (df["Stage"] == "Train")
                    & (df["D_r0"] == D_r0)
                    & (df["r0_Type"] == r0_Type)
                    & (df["Embed"] != "No")
                    & (df["Model"].isin(['GLAO', 'Direct MLNN', 'ResNet', 'Pseudo-LS', 'Pseudo-MMSE']))]
            # df0['Model'] = df0['Model'].cat.remove_unused_categories()
            df0 = df0.sort_values(by=["Model"])
            g = sns.relplot(data=df0, kind='line', 
                x='Epoch', y='Moving Average', hue="Model",
                facet_kws={'sharey': "row", 'sharex': True})
            g.set(yscale='log', ylabel="Moving Average Loss (rads$^2$)")
            g.figure.suptitle(f"D/r0 = {D_r0}")
            # g.figure.tight_layout()
            g.figure.subplots_adjust(top=0.9)

    for D_r0 in df["D_r0"].unique():
        for r0_Type in df["r0_Type"].unique():
            df0 = df[(df["Save_Epoch"] == "Final") 
                    & (df["Type"] == "Reconstruction")
                    & (df["Stage"] == "Train")
                    & (df["D_r0"] == D_r0)
                    & (df["r0_Type"] == r0_Type)
                    & (df["Embed"] != "No")
                    & (df["Model"].isin(['Pseudo-LS', 'ResNet']))]
            # df0['Model'] = df0['Model'].cat.remove_unused_categories()
            df0 = df0.sort_values(by=["Model"])
            g = sns.relplot(data=df0, kind='line', 
                x='Epoch', y='Moving Average', hue="Model", style='Size',
                facet_kws={'sharey': "row", 'sharex': True})
            g.set(yscale='log', ylabel="Moving Average Loss (rads$^2$)")
            g.figure.suptitle(f"D/r0 = {D_r0}")
            # g.figure.tight_layout()
            g.figure.subplots_adjust(top=0.9)

    for D_r0 in df["D_r0"].unique():
        for r0_Type in df["r0_Type"].unique():
            df0 = df[(df["Save_Epoch"] == "Final") 
                & (df["Size"] == "Standardised") 
                & (df["Type"] == "Reconstruction")
                & (df["Stage"] == "Verification")
                & (df["D_r0"] == D_r0)
                & (df["r0_Type"] == r0_Type)
                & (df["Embed"] != "No")]
            order = (
                df0.groupby('Model')['Loss']
                .mean()
                .sort_values()
                .index
            )
            # df0 = df0.sort_values(by=["Model"])
            fig, axs = plt.subplots(1, 1)
            sns.boxplot(data=df0, x='Model', y='Loss', ax=axs, log_scale=True, order=order)
            axs.set_title(f"D/r0 = {D_r0}")
            axs.tick_params(axis='x', rotation=30)
            axs.set_ylabel("Loss (rads$^2$)")

    fig, axs = plt.subplots(1, 1)
    df0 = df[(df["Save_Epoch"] == "Final") 
        & (df["Size"] == "Standardised") 
        & (df["Type"] == "Reconstruction")
        & (df["Stage"] == "Verification")
        & (df['Embed'] != "No")]
    sns.lineplot(data=df,
                 x='D_r0',
                 y='Loss',
                 hue='Model',
                 estimator='mean',
                 ax=axs)
    
    for D_r0 in Dr0_values:
        df1 = df[(df["Save_Epoch"] == "Final") 
            & (df["Size"] == "Standardised") 
            & (df["Type"] == "Reconstruction")
            & (df["Stage"] == "Verification")
            & (df["D_r0"] == D_r0)]

        print(f"*** Reconstruction Statistics: D/r0: {D_r0} ***")
        print(df1.groupby(["Model", "Embed"])["Loss"].describe())

    plt.show()

if __name__ == "__main__":
    main()