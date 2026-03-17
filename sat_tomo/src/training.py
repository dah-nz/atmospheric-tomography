import os, inspect, shutil

import pandas as pd
import numpy as np

from ml import ml, data
import log as logging

from tomography import asterisms, phase_screen, wavefront

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


    os.makedirs(f"{directory}/{cfg.start_time}")
    os.makedirs(f"{directory}/{cfg.start_time}/parameters")
    os.makedirs(f"{directory}/{cfg.start_time}/definitions")

    os.makedirs(f"{directory}/current", exist_ok=True)
    os.makedirs(f"{directory}/current/parameters", exist_ok=True)
    os.makedirs(f"{directory}/current/definitions", exist_ok=True)

    cfg.save(f"{directory}/current")
    cfg.save(f"{directory}/{cfg.start_time}")

    total_epochs = sum([r[0] for r in cfg.regime_settings])

    save_asterism_data = {"Sample": [], "Variable": []}

    if cfg.shwfs:
        log.info(f"Using SHWFS for coefficient estimation")
        shwfs = wavefront.SHWFS(cfg.diameter, cfg.num_modes, noise=cfg.noise)
    else:
        log.info(f"Using dot product (ideal WFS) for coefficient estimation")
        shwfs = None

    mask = cfg.basis.get_mask(cfg.aperture_size)

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
            log.info(f"Running Epoch {epoch}/{total_epochs} (Sub-epoch {e+1}/{n_epochs})")
            atmosphere_train = phase_screen.Atmosphere(n_screens, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size, per_layer_r0=cfg.per_layer_r0)
            # atmosphere_test = phase_screen.Atmosphere(cfg.testing_size, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size, per_layer_r0=cfg.per_layer_r0)

            log.info(f"Loading asterism groups. Training: {cfg.training_size}. Testing: {cfg.testing_size}")

            for asterism_set in asterism_sets:
                print(asterism_set)
                if asterism_set == "Variable":
                    training_asterisms = asterisms.generate_asterisms(cfg.training_size, cfg.magnitude, mask=mask, quality_threshold=cfg.asterism_quality_threshold, quality_limit=cfg.minimum_asterism_quality, use_regime_model=use_regime_model)
                    # testing_asterisms = asterisms.generate_asterisms(cfg.testing_size, cfg.magnitude, mask=mask, quality_threshold=cfg.asterism_quality_threshold, quality_limit=cfg.minimum_asterism_quality, use_regime_model=use_regime_model)
                    trainer = variable_trainer
                else:
                    training_asterisms = asterisms.sample_asterisms(cfg.training_size, mask=mask, seed=cfg.sample_seed)
                    # testing_asterisms = asterisms.sample_asterisms(cfg.testing_size, mask=mask, seed=cfg.sample_seed)
                    trainer = sample_trainer
                
                data = run_epoch(trainer,
                                    cfg.training_size, 
                                    training_asterisms, 
                                    atmosphere_train, 
                                    cfg.basis,
                                    wfs=shwfs)

                append_results(results, epoch, "Train", n_screens, asterism_set, data)

                # data = run_epoch(trainer, 
                #                     cfg.testing_size,
                #                     testing_asterisms,
                #                     atmosphere_test,
                #                     cfg.basis,
                #                     train=False,
                #                     wfs=shwfs)
                    
                # append_results(results, epoch, "Test", cfg.testing_size, asterism_set, data)

                if (epoch - 1) % cfg.save_frequency == 0:
                    trainer.save_models(epoch, suffix=asterism_set, dir=f"{directory}/{cfg.start_time}")
                    trainer.save_models(epoch, suffix=asterism_set, dir=f"{directory}/current")

                # save_asterism_data[asterism_set] += (epoch, np.concatenate([training_asterisms, testing_asterisms]))

            try: 
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                log.info(f"tracemalloc: {top_stats[:3]}")
            except RuntimeError:
                pass

    # Save final model
    if "Variable" in asterism_sets:
        variable_trainer.save_models(suffix='Variable_Final', dir=f"{directory}/{cfg.start_time}")
        variable_trainer.save_models(suffix='Variable_Final', dir=f"{directory}/current")
    if "Sample" in asterism_sets:
        sample_trainer.save_models(suffix='Sample_Final', dir=f"{directory}/{cfg.start_time}")
        sample_trainer.save_models(suffix='Sample_Final', dir=f"{directory}/current")

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

    verification(cfg, log, asterism_sets, trainers, mask, shwfs, results)

    df = pd.DataFrame.from_dict(results, orient="columns")

    for dir in [f"{directory}/current", f"{directory}/{cfg.start_time}"]:

        if cfg.continue_training and "current" in dir:
            df_existing = pd.read_csv(f"{dir}/results.csv")
            previous_set = df_existing['TrainingSet'].max()
            df["TrainingSet"] = previous_set + 1
            last_epoch = df_existing['Epoch'].max()
            df["Epoch"] += last_epoch
            with open(f"{dir}/results.csv", 'w') as f:
                pd.concat([df_existing, df]).to_csv(f, sep=",")
            
        else:
            df['TrainingSet'] = 0
            try:
                with open(f"{dir}/results.csv", "w") as f:
                    df.to_csv(f, sep=",")
            except PermissionError:
                with open(f"{dir}/results_{logging.time_str}.csv", "w") as f:
                    df.to_csv(f, sep=",")

        for k, m in variable_trainer.models.items():
            name = m.__class__.__name__
            source = inspect.getsource(m.__class__)
            with open(f"{dir}/definitions/{name}.def", "w") as f:
                f.write(source)

    shutil.make_archive(f'{directory}/{cfg.start_time}', 'zip', f'{directory}/{cfg.start_time}')
    shutil.rmtree(f'{directory}/{cfg.start_time}')

    return df


def verification(cfg, log, asterism_sets, trainers, mask, shwfs, results):
    from tomography import oopao, beacon
    import torch
    # does a verification run, including 
    target = beacon.Beacon(0, 0, 14, aperture_mask=mask)
    
    for asterism_set in asterism_sets:
        if asterism_set == "Variable":
            verification_asterism = asterisms.generate_asterisms(cfg.verification_size, cfg.magnitude, mask=mask, quality_threshold=cfg.asterism_quality_threshold, quality_limit=cfg.minimum_asterism_quality, use_regime_model=True)
            trainer = trainers["Variable"]
        else:
            verification_asterism = asterisms.sample_asterisms(cfg.verification_size, mask=mask, seed=cfg.sample_seed)
            trainer = trainers["Sample"]

        log.info(f"Running verification test for {asterism_set}")
        # N = 16

        atmosphere = phase_screen.Atmosphere(cfg.verification_size, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size)

        dt = data.TomographyData(cfg.verification_size, verification_asterism, atmosphere, cfg.basis, wfs=shwfs)
        out = trainer.test(dt)

        append_results(results, cfg.epochs, "Verification", cfg.verification_size, asterism_set, out)

    return results
