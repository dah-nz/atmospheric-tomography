import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

df_16_0 = pd.read_csv("./results/results_residuals_16_SNR_0.data")
df_64_0 = pd.read_csv("./results/results_residuals_64_SNR_0.data")

df = pd.concat([df_16_0, df_64_0])

palette = sns.cubehelix_palette(light=.8, n_colors=3)

sns.relplot(
    data=df, kind="line",
    x="Target Gamma", y="RMSE R:A", col="Basis Function",
    hue="N Guide Stars", style="N Coeffs",
    errorbar=None, palette=palette
)

plt.show()