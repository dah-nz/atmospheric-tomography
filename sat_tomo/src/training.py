import pandas as pd

from ml import ml, data
import log as logging

from tomography import asterisms, phase_screen

def run_epoch(trainer, n, asterisms, atmosphere, basis, train=True):
    
    dt = data.TomographyData(n, asterisms, atmosphere, basis)
    if train:
        results = trainer.train(dt)
    else:
        results = trainer.test(dt)

    return results

def add_to(results, epoch, stage, model, screens, asterism, type, loss):
    results["Epoch"].append(epoch)
    results["Stage"].append(stage)
    results["Model"].append(model)
    results["Screens"].append(screens)
    results["Asterism"].append(asterism)
    results["Type"].append(type)
    results["Loss"].append(loss)

def run(cfg):
    log = logging.get_logger(__name__)
    logging.setup_logger(log, start_time=cfg.start_time, to_console=True)
    
    fixed_trainer = ml.Trainer(cfg)
    variable_trainer = ml.Trainer(cfg)
    epoch = 0

    results = {
        "Epoch": [],
        "Stage": [],
        "Model": [],
        "Screens": [],
        "Asterism": [],
        "Type": [],
        "Loss": [],
    }

    total_epochs = sum([r[0] for r in cfg.regime])
    epoch_index = []
    n_screens_at_epoch = []

    save_asterism_data = {"fixed": [], "variable": []}

    for n_epochs, n_screens, training_size, testing_size in cfg.regime:
        for e in range(n_epochs):
            epoch += 1
            log.info(f"Running Epoch {epoch}/{total_epochs} (Sub-epoch {e+1}/{n_epochs})")
            atmosphere_train = phase_screen.Atmosphere(n_screens, cfg.altitudes, cfg.layer_conditions, cfg.D, cfg.aperture_size)
            atmosphere_test = phase_screen.Atmosphere(testing_size, cfg.altitudes, cfg.layer_conditions, cfg.D, cfg.aperture_size)

            if 'fixed' in cfg.asterism_types:
                log.info(f"Running fixed asterisms [{epoch}/{total_epochs}] ({e+1}/{n_epochs})")
                fixed_asterisms = asterisms.generate_fixed_asterisms(cfg.fixed_asterism_angle)
                data = run_epoch(fixed_trainer, 
                                 training_size, 
                                 fixed_asterisms, 
                                 atmosphere_train, 
                                 cfg.basis)
                
                for k, (l, r) in data.items():
                    add_to(results, epoch, "Train", k, n_screens, "Fixed", "Coefficients", l)
                    add_to(results, epoch, "Train", k, n_screens, "Fixed", "Reconstruction", r)

                data = run_epoch(fixed_trainer, 
                                 testing_size,
                                 fixed_asterisms,
                                 atmosphere_test,
                                 cfg.basis,
                                 train=False)
                
                for k, (l, r) in data.items():
                    add_to(results, epoch, "Test", k, testing_size, "Fixed", "Coefficients", l)
                    add_to(results, epoch, "Test", k, testing_size, "Fixed", "Reconstruction", r)
                
                save_asterism_data["fixed"] += fixed_asterisms

                fixed_trainer.save_models(epoch, suffix="fixed")
            
            if 'variable' in cfg.asterism_types:
                log.info(f"Running variable asterisms [{epoch}/{total_epochs}] ({e+1}/{n_epochs})")
                variable_asterisms = asterisms.generate_variable_asterisms(training_size + testing_size, cfg.asterism_magnitude)
                data = run_epoch(variable_trainer,
                                 training_size, 
                                 variable_asterisms, 
                                 atmosphere_train, 
                                 cfg.basis)

                for k, (l, r) in data.items():
                    add_to(results, epoch, "Train", k, n_screens, "Variable", "Coefficients", l)
                    add_to(results, epoch, "Train", k, n_screens, "Variable", "Reconstruction", r)

                data = run_epoch(variable_trainer, 
                                 testing_size,
                                 variable_asterisms,
                                 atmosphere_test,
                                 cfg.basis,
                                 train=False)
                
                for k, (l, r) in data.items():
                    add_to(results, epoch, "Test", k, testing_size, "Variable", "Coefficients", l)
                    add_to(results, epoch, "Test", k, testing_size, "Variable", "Reconstruction", r)

                variable_trainer.save_models(epoch, suffix="variable")

                save_asterism_data["variable"] += variable_asterisms

    df = pd.DataFrame.from_dict(results, orient="columns")

    with open("asterisms_variable.txt", "w") as f:
        for l in save_asterism_data["variable"]:
            f.write(f"{l}\n")

    with open("asterisms_fixed.txt", "w") as f:
        for l in save_asterism_data["fixed"]:
            f.write(f"{l}\n")

    with open("data/models/results/results.csv", "w") as f:
        df.to_csv(f, sep=",")

    with open(f"data/models/results/old/results_{cfg.start_time}.csv", "w") as f:
        df.to_csv(f, sep=",")

    return df