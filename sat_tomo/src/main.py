import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import cfg as config
import log as logging
import training
import astronomy.astronomy
import astronomy.telescope
import astronomy.satellite
import astronomy.asterisms

import display

import tracemalloc


# ... run your application ...



if __name__ == "__main__":
    tracemalloc.start()
    log = logging.get_logger(__name__)
    logging.setup_logger(log, level=logging.logging.INFO, to_console=True)

    cfg = config.Config(time=logging.time)

    tele = astronomy.telescope.default_telescope()

    if cfg.build_star_map:
        log.info(" > Load star data <")
        stars = astronomy.astronomy.StarData()

        log.info(" > Building star map <")
        time = cfg.time.astimezone(dt.timezone.utc)
        starmap = astronomy.astronomy.build_star_map(stars, cfg.magnitude, location=tele.location, time=time)

        log.info(f" > Star map completed for magnitude {cfg.magnitude} <")

    if cfg.satellites:
        log.info(" > Loading satellite information <")
        astronomy.satellite.load_sun_moon_emphemeris()
        sats = astronomy.satellite.SatelliteMap(tele, 
                                                n_sats=cfg.num_sats,
                                                star_magnitude=cfg.magnitude,
                                                time_period=cfg.period,
                                                time_delta=cfg.sat_del_t,
                                                asterism_size=cfg.asterism_size,
                                                max_magnitude=cfg.max_magnitude)
        sats.build()

    if cfg.display_asterisms:
        # print("Cleaning old data")
        # astronomy.satellite.clean_maps()
        print("Displaying asterism data")
        data = astronomy.asterisms.load_data(cfg.magnitude, limit=120, asterism_size=5)
        print(data[:5])
        print(len(data))
        # display.asterisms()

    if cfg.train_models:
        log.info(" > Start model training < ")

        training.run(cfg)

        log.info(" > Model training complete <")

    if cfg.display_training:
        # import shutil
        # shutil.unpack_archive(f"{cfg.directory}/20250409_144309.zip", f"{cfg.directory}/temp", format="zip")
        # with open(f"{cfg.directory}/temp/results.csv", "r") as f:
        #     df = pd.read_csv(f)

        with open(f"{cfg.directory}/current/results.csv", "r") as f:
            df = pd.read_csv(f)

        df_v = df.drop(df[df.Stage != "Verification"].index)
        df = df.drop(df[df.Stage == "Verification"].index)

        df = df.sort_values(by="Stage")

        sns.relplot(data=df, kind='line', 
                    x='Epoch', y='Loss', row="Type",
                    col="Asterism", style="Stage", hue="Model",
                    facet_kws={'sharey': "row", 'sharex': True})
        
        # Remove all but last verification results
        try:
            max_training_set = df_v["TrainingSet"].max()
            df_v = df_v.drop(df_v[df_v.TrainingSet != max_training_set].index)
        except KeyError:
            pass

        
        fig, axs = plt.subplots(1, 2)
        df_v_recon = df_v.drop(df_v[df_v.Type != "Reconstruction"].index)
        sns.boxplot(data=df_v_recon, x='Model', y='Loss', ax=axs[0])
        axs[0].set_title("Reconstruction")
        
        df_v_coeff = df_v.drop(df_v[df_v.Type != "Coefficients"].index)
        sns.boxplot(data=df_v_coeff, x='Model', y='Loss', ax=axs[1])
        axs[1].set_title("Coefficients")

        print("*** Reconstruction Statistics ***")
        print(df_v_recon[["Model", "Loss"]].groupby("Model").describe())

        print("*** Coefficient Statistics ***")
        print(df_v_coeff[["Model", "Loss"]].groupby("Model").describe())
        
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
        
    plt.show()
