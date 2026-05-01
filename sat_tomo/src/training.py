import os, inspect, shutil

import pandas as pd
import numpy as np

from ml import ml, data
import log as logging

from tomography import asterisms, phase_screen, wavefront, beacon
from util import var_to_r0

import tracemalloc
from tqdm import tqdm

def run_epoch(trainer, n, asterisms, atmosphere, basis, train=True, wfs=None):
    
    dt = data.TomographyData(n, asterisms, atmosphere, basis, wfs=wfs)
    if train:
        results = trainer.train(dt)
    else:
        results = trainer.test(dt)

    return results

def append_results(results, epoch, stage, screens, asterism, output_dict):
    for model, data_dict in output_dict.items():
        for key, value in data_dict.items():
            if stage != "Verification":

                results["Epoch"].append(epoch)
                results["Stage"].append(stage)
                results["Model"].append(model)
                results["Screens"].append(screens)
                results["Asterism"].append(asterism)
                results["Type"].append(key)
                results["Loss"].append(np.mean(value))
            else:
                for v in value:
                    results["Epoch"].append(epoch)
                    results["Stage"].append(stage)
                    results["Model"].append(model)
                    results["Screens"].append(screens)
                    results["Asterism"].append(asterism)
                    results["Type"].append(key)
                    results["Loss"].append(v)

def run(cfg):
    log = logging.get_logger(__name__)
    logging.setup_logger(log, to_console=True)
    
    variable_trainer = ml.Trainer(cfg, skip_lsq=True)
    if not cfg.sample_set_off:
        sample_trainer = ml.Trainer(cfg, skip_lsq=False)
    
    directory = cfg.directory

    if cfg.load_previous or cfg.continue_training:
        variable_trainer.load_models(dir=f"{directory}/current", suffix="Variable_Final")
        if not cfg.sample_set_off:
            sample_trainer.load_models(dir=f"{directory}/current", suffix="Sample_Final")

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

    phase_data = {
        "Epoch": [],
        "Index": [],
        "Layer": [],
        "Variance": [],
        "r0": []
    }

    asterism_data = {
        "Epoch": [],
        "Index": [],
        "Star": [],
        "Theta": [],
        "Gamma": [],
        "Magnitude": []
    }

    model_data = []

    os.makedirs(f"{directory}/{cfg.start_time}")
    os.makedirs(f"{directory}/{cfg.start_time}/parameters")
    os.makedirs(f"{directory}/{cfg.start_time}/definitions")

    os.makedirs(f"{directory}/current", exist_ok=True)
    os.makedirs(f"{directory}/current/parameters", exist_ok=True)
    os.makedirs(f"{directory}/current/definitions", exist_ok=True)

    cfg.save(f"{directory}/current")
    cfg.save(f"{directory}/{cfg.start_time}")

    total_epochs = sum([r[0] for r in cfg.regime_settings])

    if cfg.shwfs:
        log.info(f"Using SHWFS for coefficient estimation")
        shwfs = wavefront.SHWFS(cfg.diameter, cfg.num_modes, noise=cfg.noise)
    else:
        log.info(f"Using dot product (ideal WFS) for coefficient estimation")
        shwfs = None

    mask = cfg.basis.get_mask(cfg.aperture_size)
    target = beacon.Beacon(0, 0, 0, aperture_mask=mask)

    asterisms.LIMITING_ANGLE = cfg.limiting_angle
    asterism_sets = ['Variable']
    if not cfg.sample_set_off:
        asterism_sets.append('Sample')

    if cfg.regime == "quality_split":
        regime_settings = cfg.regime_settings
    else:
        regime_settings = [(a, b, True) for a,b in cfg.regime_settings]

    for n_epochs, n_screens, use_regime_model in regime_settings:
        for e in range(n_epochs):
            epoch += 1
            log.info(f"Running Epoch {epoch}/{total_epochs} (Sub-epoch {e+1}/{n_epochs}) [{cfg.r0}]")
            atmosphere_train = phase_screen.Atmosphere(n_screens, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size)

            for n in range(n_screens):
                for layer in range(len(cfg.altitudes)):
                    var = atmosphere_train.generators[layer].screen_var[n]
                    r0 = atmosphere_train.generators[layer].screen_r0[n]

                    phase_data["Epoch"].append(epoch)
                    phase_data["Index"].append(n)
                    phase_data["Layer"].append(layer+1)
                    phase_data["r0"].append(r0)
                    phase_data["Variance"].append(var)

                co = target.total_cutout(cfg.diameter, cfg.altitudes, atmosphere_train[n], atmosphere_train.metapupil_diameters)
                var = np.var(co)
                r0 = var_to_r0(cfg.diameter, var)

                phase_data["Epoch"].append(epoch)
                phase_data["Index"].append(n)
                phase_data["Layer"].append("Total")
                phase_data["r0"].append(r0)
                phase_data["Variance"].append(var)
                

            log.info(f"Loading asterism groups. Training: {cfg.training_size}. Testing: {cfg.testing_size}")

            for asterism_set in asterism_sets:
                print(asterism_set)
                if asterism_set == "Variable":
                    training_asterisms = asterisms.generate_asterisms(cfg.training_size, cfg.magnitude, mask=mask, quality_threshold=cfg.asterism_quality_threshold, quality_limit=cfg.minimum_asterism_quality, use_regime_model=use_regime_model)
                    trainer = variable_trainer
                else:
                    training_asterisms = asterisms.sample_asterisms(cfg.training_size, mask=mask, seed=cfg.sample_seed)
                    trainer = sample_trainer

                for n in range(cfg.training_size):
                    for s in range(cfg.asterism_size):
                        asterism_data["Epoch"].append(epoch)
                        asterism_data["Index"].append(n)
                        asterism_data["Star"].append(s)
                        asterism_data["Gamma"].append(training_asterisms[n][s]._gamma_arcsec)
                        asterism_data["Theta"].append(training_asterisms[n][s].theta)
                        asterism_data["Magnitude"].append(training_asterisms[n][s].magnitude)
                
                data = run_epoch(trainer,
                                    cfg.training_size, 
                                    training_asterisms, 
                                    atmosphere_train, 
                                    cfg.basis,
                                    wfs=shwfs)

                append_results(results, epoch, "Train", n_screens, asterism_set, data)

                if (epoch - 1) % cfg.save_frequency == 0:
                    model_files = trainer.save_models(epoch, suffix=f"_{asterism_set}", dir=f"{directory}/{cfg.start_time}")
                    trainer.save_models(epoch, suffix=f"_{asterism_set}", dir=f"{directory}/current")

                    for model_path in model_files:
                        model_data.append(model_path)

            try: 
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                log.info(f"tracemalloc: {top_stats[:3]}")
            except RuntimeError:
                pass

    # Save final model
    if "Variable" in asterism_sets:
        model_files = variable_trainer.save_models(suffix='_Variable_Final', dir=f"{directory}/{cfg.start_time}")
        variable_trainer.save_models(suffix='_Variable_Final', dir=f"{directory}/current")

        for model_path in model_files:
            model_data.append(model_path)

    if "Sample" in asterism_sets:
        model_files = sample_trainer.save_models(suffix='_Sample_Final', dir=f"{directory}/{cfg.start_time}")
        sample_trainer.save_models(suffix='_Sample_Final', dir=f"{directory}/current")
        for model_path in model_files:
            model_data.append(model_path)

    # Run verification test
    log.info(f"Running final verification test")
    for runner in variable_trainer.runners:
        try:
            runner.skip_lsq = False
        except AttributeError:
            pass

    # PERFORM VERIFICATION
    if not cfg.sample_set_off:
        trainers = {
            "Variable": variable_trainer,
            "Sample": sample_trainer
        }
    else:
        trainers = {
            "Variable": variable_trainer,
        }

    results, phase_data, asterism_data = verification(cfg, log, asterism_sets, model_data, mask, shwfs, results, phase_data, asterism_data)

    df = pd.DataFrame.from_dict(results, orient="columns")
    df_phase = pd.DataFrame.from_dict(phase_data, orient="columns")
    df_asterism = pd.DataFrame.from_dict(asterism_data, orient="columns")

    for dir in [f"{directory}/current", f"{directory}/{cfg.start_time}"]:

        if cfg.continue_training and "current" in dir:
            df_existing = pd.read_csv(f"{dir}/results.csv")
            previous_set = df_existing['TrainingSet'].max()
            df["TrainingSet"] = previous_set + 1
            last_epoch = df_existing['Epoch'].max()
            df["Epoch"] += last_epoch
            with open(f"{dir}/results.csv", 'w') as f:
                pd.concat([df_existing, df]).to_csv(f, sep=",")

            df_phase.to_csv(f"{dir}/phase_{previous_set+1}.csv")
            df_asterism.to_csv(f"{dir}/asterisms_{previous_set+1}.csv")
            
        else:
            df['TrainingSet'] = 0
            try:
                with open(f"{dir}/results.csv", "w") as f:
                    df.to_csv(f, sep=",")
            except PermissionError:
                with open(f"{dir}/results_{logging.time_str}.csv", "w") as f:
                    df.to_csv(f, sep=",")

            df_phase.to_csv(f"{dir}/phase.csv")
            df_asterism.to_csv(f"{dir}/asterisms.csv")

        for k, m in variable_trainer.models.items():
            name = m.__class__.__name__
            source = inspect.getsource(m.__class__)
            with open(f"{dir}/definitions/{name}.def", "w") as f:
                f.write(source)

    shutil.make_archive(f'{directory}/{cfg.start_time}', 'zip', f'{directory}/{cfg.start_time}')
    shutil.rmtree(f'{directory}/{cfg.start_time}')

    return df


def verification(cfg, log, asterism_sets, model_data, mask, shwfs, results, phase_data, asterism_data):
    from tomography import oopao, beacon
    import torch
    # does a verification run, including 
    target = beacon.Beacon(0, 0, 14, aperture_mask=mask)

    trainer = ml.Trainer(cfg, model_dict=model_data)
    trainer.load_models(model_dict=model_data)
    
    for asterism_set in asterism_sets:
        if asterism_set == "Variable":
            verification_asterism = asterisms.generate_asterisms(cfg.verification_size, cfg.magnitude, mask=mask, quality_threshold=cfg.asterism_quality_threshold, quality_limit=cfg.minimum_asterism_quality, use_regime_model=True)
        else:
            verification_asterism = asterisms.sample_asterisms(cfg.verification_size, mask=mask, seed=cfg.sample_seed)

        for n in range(cfg.verification_size):
            for s in range(cfg.asterism_size):
                asterism_data["Epoch"].append("Verification")
                asterism_data["Index"].append(n)
                asterism_data["Star"].append(s)
                asterism_data["Gamma"].append(verification_asterism[n][s]._gamma_arcsec)
                asterism_data["Theta"].append(verification_asterism[n][s].theta)
                asterism_data["Magnitude"].append(verification_asterism[n][s].magnitude)

        log.info(f"Running verification test for {asterism_set}")
        # N = 16

        atmosphere = phase_screen.Atmosphere(cfg.verification_size, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size)

        for n in range(cfg.verification_size):
            for layer in range(len(cfg.altitudes)):
                var = atmosphere.generators[layer].screen_var[n]
                r0 = atmosphere.generators[layer].screen_r0[n]

                phase_data["Epoch"].append("Verification")
                phase_data["Index"].append(n)
                phase_data["Layer"].append(layer+1)
                phase_data["r0"].append(r0)
                phase_data["Variance"].append(var)

            co = target.total_cutout(cfg.diameter, cfg.altitudes, atmosphere[n], atmosphere.metapupil_diameters)
            var = np.var(co)
            r0 = var_to_r0(cfg.diameter, var)

            phase_data["Epoch"].append("Verification")
            phase_data["Index"].append(n)
            phase_data["Layer"].append("Total")
            phase_data["r0"].append(r0)
            phase_data["Variance"].append(var)

        dt = data.TomographyData(cfg.verification_size, verification_asterism, atmosphere, cfg.basis, wfs=shwfs)
        out = trainer.test(dt)

        append_results(results, cfg.epochs, "Verification", cfg.verification_size, asterism_set, out)

    return results, phase_data, asterism_data
