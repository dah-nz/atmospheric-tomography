import os, inspect

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

def append_results(results, epoch, stage, model, screens, asterism, type, loss):
    results["Epoch"].append(epoch)
    results["Stage"].append(stage)
    results["Model"].append(model)
    results["Screens"].append(screens)
    results["Asterism"].append(asterism)
    results["Type"].append(type)
    results["Loss"].append(loss)

def run(cfg):
    log = logging.get_logger(__name__)
    logging.setup_logger(log, to_console=True)
    
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

    directory = "data/models"
    os.makedirs(f"{directory}/{cfg.start_time}")
    os.makedirs(f"{directory}/{cfg.start_time}/parameters")
    os.makedirs(f"{directory}/{cfg.start_time}/definitions")

    total_epochs = sum([r[0] for r in cfg.regime])

    save_asterism_data = {"fixed": [], "variable": []}

    for n_epochs, n_screens in cfg.regime:
        for e in range(n_epochs):
            epoch += 1
            log.info(f"Running Epoch {epoch}/{total_epochs} (Sub-epoch {e+1}/{n_epochs})")
            atmosphere_train = phase_screen.Atmosphere(n_screens, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size)
            atmosphere_test = phase_screen.Atmosphere(cfg.testing_size, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size)

            if 'fixed' in cfg.asterism_types:
                log.info(f"Running fixed asterisms [{epoch}/{total_epochs}] ({e+1}/{n_epochs})")
                fixed_asterisms = asterisms.generate_fixed_asterisms(cfg.fixed_asterism_angle)
                data = run_epoch(fixed_trainer, 
                                 cfg.training_size, 
                                 fixed_asterisms, 
                                 atmosphere_train, 
                                 cfg.basis)
                
                for k, (l, r) in data.items():
                    append_results(results, epoch, "Train", k, n_screens, "Fixed", "Coefficients", l)
                    append_results(results, epoch, "Train", k, n_screens, "Fixed", "Reconstruction", r)

                data = run_epoch(fixed_trainer, 
                                 cfg.testing_size,
                                 fixed_asterisms,
                                 atmosphere_test,
                                 cfg.basis,
                                 train=False)
                
                for k, (l, r) in data.items():
                    append_results(results, epoch, "Test", k, cfg.testing_size, "Fixed", "Coefficients", l)
                    append_results(results, epoch, "Test", k, cfg.testing_size, "Fixed", "Reconstruction", r)
                
                save_asterism_data["fixed"] += (epoch, fixed_asterisms)

                fixed_trainer.save_models(epoch, suffix="fixed", dir=f"{directory}/{cfg.start_time}")
                fixed_trainer.save_models(epoch, suffix="fixed", dir=f"{directory}/current")
            
            if 'variable' in cfg.asterism_types:
                log.info(f"Running variable asterisms [{epoch}/{total_epochs}] ({e+1}/{n_epochs})")
                variable_asterisms = asterisms.generate_variable_asterisms(cfg.training_size + cfg.testing_size, cfg.magnitude)
                data = run_epoch(variable_trainer,
                                 cfg.training_size, 
                                 variable_asterisms, 
                                 atmosphere_train, 
                                 cfg.basis)

                for k, (l, r) in data.items():
                    append_results(results, epoch, "Train", k, n_screens, "Variable", "Coefficients", l)
                    append_results(results, epoch, "Train", k, n_screens, "Variable", "Reconstruction", r)

                data = run_epoch(variable_trainer, 
                                 cfg.testing_size,
                                 variable_asterisms,
                                 atmosphere_test,
                                 cfg.basis,
                                 train=False)
                
                for k, (l, r) in data.items():
                    append_results(results, epoch, "Test", k, cfg.testing_size, "Variable", "Coefficients", l)
                    append_results(results, epoch, "Test", k, cfg.testing_size, "Variable", "Reconstruction", r)

                variable_trainer.save_models(epoch, suffix="variable", dir=f"{directory}/{cfg.start_time}")
                variable_trainer.save_models(epoch, suffix="variable", dir=f"{directory}/current")

                save_asterism_data["variable"] += (epoch, variable_asterisms)

    df = pd.DataFrame.from_dict(results, orient="columns")

    for dir in [f"{directory}/current", f"{directory}/{cfg.start_time}"]:
        with open(f"{dir}/asterisms_variable.txt", "w") as f:
            for l in save_asterism_data["variable"]:
                f.write(f"{l}\n")

        with open(f"{dir}/asterisms_fixed.txt", "w") as f:
            for l in save_asterism_data["fixed"]:
                f.write(f"{l}\n")

        with open(f"{dir}/results.csv", "w") as f:
            df.to_csv(f, sep=",")

        for k, m in fixed_trainer.models.items():
            name = m.__class__.__name__
            source = inspect.getsource(m.__class__)
            with open(f"{dir}/definitions/{name}.def", "w") as f:
                f.write(source)

    return df