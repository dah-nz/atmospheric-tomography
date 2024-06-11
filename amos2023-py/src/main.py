import os, sys
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from util import log

from ml import ml

logger = log.get_logger(__name__)

N_BEST = 3

def main():
    log.setup_logger(logger)

    logger.info("Started")

    models = update_model_list()

    models_todo = list(models.keys())
    models_trained = []

    while len(models_todo) > 0:
        logger.info(f"Models to be trained: {len(models_todo)}. Models completed: {len(models_trained)}.")
        m = models_todo.pop()
        logger.info(f"Training model: {m}")
        ml.train(models[m])
        models_trained.append(m)

        # Check if new model definitions are in the directory
        models = update_model_list(models)
        for m in models.keys():
            if m not in models_todo and m not in models_trained:
                logger.info(f"Loaded new model: {m}")
                models_todo.append(m)

    logger.info(f"Comparing models")
    rankings = ml.compare(models)

    logger.info(f"Displaying model data")
    ml.display(models, rankings)

    clear(models, rankings)

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
    
def clear(q, rankings):
    # Clear old models and results from the workspace directories
    # Does not change the values in q
    start_time = log.time
    n_models = len(list(q.keys()))

    logger.info(f"Clearing model files")
    if n_models > N_BEST:
        for n, (name, performance, file_name) in enumerate(rankings):
            if n >= N_BEST:
                logger.info(f"Model: {name} is not one of the top {N_BEST}, moving definition file to old")
                def0 = f"data/models/definitions/{name}.def"
                def1 = f"data/models/definitions/old/{name}_{start_time}.def"
                os.rename(def0, def1)
            else:
                logger.info(f"Model: {name} is one of the top {N_BEST}")
    else:
        logger.info(f"There are {n_models}/{N_BEST}. None are moved")
        
    models = f"data/models/params/*.mdl"
    files = glob.glob(models)
    for file in files:
        f = file.split('\\')
        nm = f[1].split(".")
        f2 = f"{f[0]}/old/{nm[0]}_{start_time}.{nm[1]}"
        os.rename(file, f2)

    results = f"data/models/results/*.dat"
    files = glob.glob(results)
    for file in files:
        f = file.split('\\')
        nm = f[1].split(".")
        f2 = f"{f[0]}/old/{nm[0]}_{start_time}.{nm[1]}"
        os.rename(file, f2)

if __name__ == "__main__":
    main()