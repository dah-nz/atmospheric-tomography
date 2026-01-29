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

    print(df_gs.head())
    g = sns.relplot(data=df_gs, kind='line', 
                x='Epoch', y='Moving Average', row="Type",
                style="Asterism", hue="Model",
                facet_kws={'sharey': "row", 'sharex': True})
    g.figure.suptitle(f"Guidestar Reconstruction Loss")

    print(df_long.head())
    for comparison in ['RseNet']: #, 'MLNN']:
        df = df_long[~df_long['Model'].str.contains(comparison)]

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

    
    fig, axs = plt.subplots(1, 1)
    df_v_recon = df_v.drop(df_v[df_v.Type != "Reconstruction"].index)
    sns.boxplot(data=df_v_recon, x='Model', y='Loss', ax=axs)
    axs.set_title("Reconstruction")
    axs.tick_params(axis='x', rotation=30)
    
    # df_v_coeff = df_v.drop(df_v[df_v.Type != "Coefficients"].index)
    # sns.boxplot(data=df_v_coeff, x='Model', y='Loss', ax=axs[1])
    # axs[1].set_title("Coefficients")
    # axs[1].tick_params(axis='x', rotation=30)

    print("*** Reconstruction Statistics ***")
    print(df_v_recon[["Model", "Loss"]].groupby("Model").describe())

    # print("*** Coefficient Statistics ***")
    # print(df_v_coeff[["Model", "Loss"]].groupby("Model").describe())
    
