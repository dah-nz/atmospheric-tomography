import cfg as config
import log as logging
import training

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    cfg = config.Config()

    log = logging.get_logger(__name__)
    logging.setup_logger(log, start_time=cfg.start_time, to_console=True)

    log.info(" > Start model training < ")

    training.run(cfg)

    log.info(" > Model training complete <")

    with open("data/models/results/results.csv", "r") as f:
        df = pd.read_csv(f)

    df = df.sort_values(by="Stage")

    sns.relplot(data=df, kind='line', 
                x='Epoch', y='Loss', row="Type",
                col="Asterism", style="Stage", hue="Model",
                facet_kws={'sharey': "row", 'sharex': True})
        
    plt.show()

