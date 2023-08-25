import pandas as pd

def load():
    df_64_0_z =     pd.read_csv("./results/results_230824c_64_SNR_0_z.data")
    df_64_0_dct =   pd.read_csv("./results/results_230824c_64_SNR_0_dct.data")
    df_64_0_dst =   pd.read_csv("./results/results_230824c_64_SNR_0_dst.data")
    df_64_0_wd =    pd.read_csv("./results/results_230824c_64_SNR_0_wd.data")
    df_64_0_wh =    pd.read_csv("./results/results_230824c_64_SNR_0_wh.data")
    df_16_0_z =     pd.read_csv("./results/results_230824c_16_SNR_0_z.data")
    df_16_0_dct =   pd.read_csv("./results/results_230824c_16_SNR_0_dct.data")
    df_16_0_dst =   pd.read_csv("./results/results_230824c_16_SNR_0_dst.data")
    df_16_0_wd =    pd.read_csv("./results/results_230824c_16_SNR_0_wd.data")
    df_16_0_wh =    pd.read_csv("./results/results_230824c_16_SNR_0_wh.data")
    df_16_0 = pd.concat([df_16_0_z, df_16_0_dct, df_16_0_dst, df_16_0_wd, df_16_0_wh])
    df_64_0 = pd.concat([df_64_0_z, df_64_0_dct, df_64_0_dst, df_64_0_wd, df_64_0_wh])
    df = pd.concat([df_16_0, df_64_0])
    
    return df

def load_noise():
    dfs = []
    for n in [0, 3, 6, 10, 13, 16, 20, 23, 26, 30]:
        for b in ['z', 'dct', 'dst', 'wh', 'wd']:
            dfs.append(pd.read_csv("./results/results_noise_16_SNR_{}_{}.data".format(n, b)))
            dfs.append(pd.read_csv("./results/results_noise_64_SNR_{}_{}.data".format(n, b)))
    
    return pd.concat(dfs)
