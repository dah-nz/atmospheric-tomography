import os, sys
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from util import log
import shutil

from ml import ml
from basis import zernike

log.default_logging_level = log.logging.DEBUG
logger = log.get_logger(__name__)

N_BEST = 3

def summarise(m):
    pass
    # print(ml.model_summary(ml.Trainer(m)))

def main():
    log.setup_logger(logger)
    logger.info("Started")

    clear_previous()

    models = update_model_list()

    for _, m in models.items():
        summarise(m)

    models_todo = list(models.keys())
    models_trained = []

    # n_modes, angle, r0, diameter
    test_params = [
        # (16, 15, 0.03),
        # (16, 15, 0.05),
        # (16, 15, 0.07),
        # (16, 15, 0.09),
        # (16, 10, 0.07),
        # (16, 20, 0.07),
        # (16, 25, 0.07),
        (25, 15, 0.07),
        (32, 15, 0.07),
    ]
    
    for n_modes, theta, r0 in test_params:
        m = models_todo[0]
        logger.info(f"Training model: {m} with {n_modes} modes, {theta}\" asterism, r0 = {r0}m")

        # set up test parameters
        ml.settings["n_coeffs"] = n_modes
        ml.settings["fixed_asterism_angle"] = theta
        ml.settings["layer_conditions"] = [r0, r0, r0]
        ml.settings["basis"] = zernike.Zernike(n_modes)
        models = update_model_list()

        ml.train(models[m])
    
    ml.display()

    plt.show()
    return

def update_model_list(existing=None):
    directory = "data/models/definitions/*.def"
    files = glob.glob(directory)
    if existing is None:
        existing = {}
    for file in files:
        name, model = ml.load(file)
        if name not in existing.keys():
            logger.debug(f"Loaded new model: {name}")
            existing[name] = model

    n = len(list(existing.keys()))
    logger.debug(f"Model definition count: {n}")

    return existing

def clear_previous():
    start_time = log.time

    results = f"data/models/results/*.dat"
    files = glob.glob(results)
    if len(files) > 0:
        os.makedirs(f"data/models/results/{start_time}")
    for file in files:
        f = file.split('\\')
        f2 = f"{f[0]}/{start_time}/{f[1]}"
        os.rename(file, f2)

    models = f"data/models/params/*.mdl"
    files = glob.glob(models)
    if len(files) > 0:
        os.makedirs(f"data/models/params/{start_time}")
    for file in files:
        f = file.split('\\')
        f2 = f"{f[0]}/{start_time}/{f[1]}"
        os.rename(file, f2)

def clear(q, rankings):
    # Clear old models and results from the workspace directories
    # Does not change the values in q
    start_time = log.time
    n_models = len(list(q.keys()))

    logger.info(f"Clearing model files")
    if n_models > N_BEST:
        for n, (name, performance, file_name) in enumerate(rankings):
            if name != "Open Loop":
                def0 = f"data/models/definitions/{name}.def"
                def1 = f"data/models/definitions/old/{name}_{start_time}.def"
                if n >= N_BEST:
                    logger.info(f"Model: {name} is not one of the top {N_BEST}. Moving definition file to old")
                    os.rename(def0, def1)
                else:
                    logger.info(f"Model: {name} is one of the top {N_BEST}. Copying definition file to old")
                    shutil.copyfile(def0, def1)
    else:
        logger.info(f"There are {n_models}/{N_BEST}. None are moved")


if __name__ == "__main__":
    args = sys.argv
    if "--show" in args or '-s' in args:
        ml.display()
        plt.show()
    else:
        main()