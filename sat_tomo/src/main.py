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

if __name__ == "__main__":
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
                                                asterism_size=cfg.asterism_size)
        sats.build()

    if cfg.display_asterisms:
        print("Displaying asterism data")
        # astronomy.satellite.clean_maps()

        for magnitude in range(6, 14 + 1):
            angles = astronomy.satellite.load_map(magnitude)

            magnitudes = []
            arcseconds = []

            for angle in angles:
                for (a, _), m in angle:
                    arcseconds.append(a)
                    magnitudes.append(m)

            plt.scatter(magnitude, arcseconds)
        # print(angles[0])

    if cfg.train_models:
        log.info(" > Start model training < ")

        training.run(cfg)

        log.info(" > Model training complete <")

    if cfg.display_training:
        with open("data/models/results/results.csv", "r") as f:
            df = pd.read_csv(f)

        df = df.sort_values(by="Stage")

        sns.relplot(data=df, kind='line', 
                    x='Epoch', y='Loss', row="Type",
                    col="Asterism", style="Stage", hue="Model",
                    facet_kws={'sharey': "row", 'sharex': True})
        
    plt.show()

# Build KD trees for star distances using star map
def build_star_maps():
    from astronomy.astronomy import build_maps