import pandas as pd

def load():
    df_64_0_z =     pd.read_csv("./results/results_230823_64_SNR_0_z.data")
    df_64_0_dct =   pd.read_csv("./results/results_230823_64_SNR_0_dct.data")
    df_64_0_dst =   pd.read_csv("./results/results_230823_64_SNR_0_dst.data")
    df_64_0_wd =    pd.read_csv("./results/results_230823_64_SNR_0_wd.data")
    df_64_0_wh =    pd.read_csv("./results/results_230823_64_SNR_0_wh.data")
    df_16_0_z =     pd.read_csv("./results/results_230823_16_SNR_0_z.data")
    df_16_0_dct =   pd.read_csv("./results/results_230823_16_SNR_0_dct.data")
    df_16_0_dst =   pd.read_csv("./results/results_230823_16_SNR_0_dst.data")
    df_16_0_wd =    pd.read_csv("./results/results_230823_16_SNR_0_wd.data")
    df_16_0_wh =    pd.read_csv("./results/results_230823_16_SNR_0_wh.data")
    df_16_0 = pd.concat([df_16_0_z, df_16_0_dct, df_16_0_dst, df_16_0_wd, df_16_0_wh])
    df_64_0 = pd.concat([df_64_0_z, df_64_0_dct, df_64_0_dst, df_64_0_wd, df_64_0_wh])
    df = pd.concat([df_16_0, df_64_0])
    df = df[df["Asterism"] != 5]
    df.loc[df["Asterism"] == 3, "Asterism"] = "3_30"
    df.loc[df["Asterism"] == 6, "Asterism"] = "6_60"
    df.loc[df["Asterism"] == 1, "Asterism"] = "1_0"

    df_64_0_z =     pd.read_csv("./results/results_230824_64_SNR_0_z.data")
    df_64_0_dct =   pd.read_csv("./results/results_230824_64_SNR_0_dct.data")
    df_64_0_dst =   pd.read_csv("./results/results_230824_64_SNR_0_dst.data")
    df_64_0_wd =    pd.read_csv("./results/results_230824_64_SNR_0_wd.data")
    df_64_0_wh =    pd.read_csv("./results/results_230824_64_SNR_0_wh.data")
    df_16_0_z =     pd.read_csv("./results/results_230824_16_SNR_0_z.data")
    df_16_0_dct =   pd.read_csv("./results/results_230824_16_SNR_0_dct.data")
    df_16_0_dst =   pd.read_csv("./results/results_230824_16_SNR_0_dst.data")
    df_16_0_wd =    pd.read_csv("./results/results_230824_16_SNR_0_wd.data")
    df_16_0_wh =    pd.read_csv("./results/results_230824_16_SNR_0_wh.data")
    df_16_0 = pd.concat([df_16_0_z, df_16_0_dct, df_16_0_dst, df_16_0_wd, df_16_0_wh])
    df_64_0 = pd.concat([df_64_0_z, df_64_0_dct, df_64_0_dst, df_64_0_wd, df_64_0_wh])
    df_3s = pd.concat([df_16_0, df_64_0])
    df_3s.loc[df_3s["Asterism"] == 3, "Asterism"] = "3_15"

    df_64_0_z =     pd.read_csv("./results/results_230824b_64_SNR_0_z.data")
    df_64_0_dct =   pd.read_csv("./results/results_230824b_64_SNR_0_dct.data")
    df_64_0_dst =   pd.read_csv("./results/results_230824b_64_SNR_0_dst.data")
    df_64_0_wd =    pd.read_csv("./results/results_230824b_64_SNR_0_wd.data")
    df_64_0_wh =    pd.read_csv("./results/results_230824b_64_SNR_0_wh.data")
    df_16_0_z =     pd.read_csv("./results/results_230824b_16_SNR_0_z.data")
    df_16_0_dct =   pd.read_csv("./results/results_230824b_16_SNR_0_dct.data")
    df_16_0_dst =   pd.read_csv("./results/results_230824b_16_SNR_0_dst.data")
    df_16_0_wd =    pd.read_csv("./results/results_230824b_16_SNR_0_wd.data")
    df_16_0_wh =    pd.read_csv("./results/results_230824b_16_SNR_0_wh.data")
    df_16_0 = pd.concat([df_16_0_z, df_16_0_dct, df_16_0_dst, df_16_0_wd, df_16_0_wh])
    df_64_0 = pd.concat([df_64_0_z, df_64_0_dct, df_64_0_dst, df_64_0_wd, df_64_0_wh])
    df_5a = pd.concat([df_16_0, df_64_0])
    df_5a.loc[df_5a["Asterism"] == 5, "Asterism"] = "5_Asym"
    return pd.concat([df_3s, df, df_5a])

def load_noise():
    dfs = []
    for n in [0, 3, 6, 10, 13, 16, 20, 23, 26, 30]:
        for b in ['z', 'dct', 'dst', 'wh', 'wd']:
            dfs.append(pd.read_csv("./results/results_noise_16_SNR_{}_{}.data".format(n, b)))
            dfs.append(pd.read_csv("./results/results_noise_64_SNR_{}_{}.data".format(n, b)))
    
    return pd.concat(dfs)
