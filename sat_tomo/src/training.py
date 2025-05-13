import os, inspect, shutil

import pandas as pd
import numpy as np

from ml import ml, data
import log as logging

from tomography import asterisms, phase_screen, wavefront

import tracemalloc

def run_epoch(trainer, n, asterisms, atmosphere, basis, train=True, wfs=None, tiptilt_removal=False):
    
    dt = data.TomographyData(n, asterisms, atmosphere, basis, wfs=wfs, tiptilt_removal=tiptilt_removal)
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
    
    if cfg.train_combined:
        trainer = ml.Trainer(cfg)
    else:
        fixed_trainer = ml.Trainer(cfg)
        variable_trainer = ml.Trainer(cfg)

    directory = cfg.directory

    if cfg.load_previous or cfg.continue_training:
        if cfg.train_combined:
            trainer.load_models(dir=f"{directory}/current")
        else:
            fixed_trainer.load_models(suffix="fixed", dir=f"{directory}/current")
            variable_trainer.load_models(suffix="variable", dir=f"{directory}/current")

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

    total_epochs = sum([r[0] for r in cfg.regime])

    save_asterism_data = {"fixed": [], "variable": []}

    if cfg.shwfs:
        log.info(f"Using SHWFS for coefficient estimation")
        shwfs = wavefront.SHWFS(cfg.diameter, cfg.num_modes, noise=cfg.noise)
    else:
        log.info(f"Using dot product (ideal WFS) for coefficient estimation")
        shwfs = None

    mask = cfg.basis.get_mask(cfg.aperture_size)

    fixed_bias_c = []
    fixed_bias_r = []

    var_bias_c = []
    var_bias_r = []

    asterisms.LIMITING_ANGLE = cfg.limiting_angle

    for n_epochs, n_screens in cfg.regime:
        for e in range(n_epochs):
            epoch += 1
            log.info(f"Running Epoch {epoch}/{total_epochs} (Sub-epoch {e+1}/{n_epochs})")
            atmosphere_train = phase_screen.Atmosphere(n_screens, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size, per_layer_r0=cfg.per_layer_r0)
            atmosphere_test = phase_screen.Atmosphere(cfg.testing_size, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size, per_layer_r0=cfg.per_layer_r0)

            if 'fixed' in cfg.asterism_types:
                log.info(f"Running fixed asterisms [{epoch}/{total_epochs}] ({e+1}/{n_epochs})")
                fixed_asterisms_train = asterisms.generate_fixed_asterisms(cfg.training_size, cfg.fixed_asterism_angle, mask=mask, theta_0_random=cfg.fixed_theta_0_randomised)
                if cfg.train_combined:
                    fixed_trainer = trainer
                data = run_epoch(fixed_trainer, 
                                 cfg.training_size, 
                                 fixed_asterisms_train, 
                                 atmosphere_train, 
                                 cfg.basis,
                                 wfs=shwfs,
                                 tiptilt_removal=cfg.wavefront_tip_tilt_removal)
                
                for k, (l, r, bc, br) in data.items():
                    append_results(results, epoch, "Train", k, n_screens, "Fixed", "Coefficients", l)
                    append_results(results, epoch, "Train", k, n_screens, "Fixed", "Reconstruction", r)
                    fixed_bias_c.append(bc)
                    fixed_bias_r.append(br)


                fixed_asterisms_test = asterisms.generate_fixed_asterisms(cfg.testing_size, cfg.fixed_asterism_angle, mask=mask, theta_0_random=cfg.fixed_theta_0_randomised)
                data = run_epoch(fixed_trainer, 
                                 cfg.testing_size,
                                 fixed_asterisms_test,
                                 atmosphere_test,
                                 cfg.basis,
                                 train=False,
                                 wfs=shwfs,
                                 tiptilt_removal=cfg.wavefront_tip_tilt_removal)

                for k, (l, r, _, _) in data.items():
                    append_results(results, epoch, "Test", k, cfg.testing_size, "Fixed", "Coefficients", l)
                    append_results(results, epoch, "Test", k, cfg.testing_size, "Fixed", "Reconstruction", r)
                
                save_asterism_data["fixed"] += (epoch, fixed_asterisms_train + fixed_asterisms_test)

                if (epoch - 1) % cfg.save_frequency == 0 and not cfg.train_combined:
                    fixed_trainer.save_models(epoch, suffix="fixed", dir=f"{directory}/{cfg.start_time}")
                    fixed_trainer.save_models(epoch, suffix="fixed", dir=f"{directory}/current")
            
            if 'variable' in cfg.asterism_types:
                log.info(f"Running variable asterisms [{epoch}/{total_epochs}] ({e+1}/{n_epochs})")
                training_asterisms = asterisms.generate_variable_asterisms(cfg.training_size, cfg.magnitude, zero_rate=1e-3, fixed_rate=1e-3, mask=mask)
                testing_asterisms = asterisms.generate_variable_asterisms(cfg.testing_size, cfg.magnitude, zero_rate=0.0, fixed_rate=0.0, mask=mask)
                if cfg.train_combined:
                    variable_trainer = trainer

                data = run_epoch(variable_trainer,
                                 cfg.training_size, 
                                 training_asterisms, 
                                 atmosphere_train, 
                                 cfg.basis,
                                 wfs=shwfs,
                                 tiptilt_removal=cfg.wavefront_tip_tilt_removal)

                for k, (l, r, bc, br) in data.items():
                    append_results(results, epoch, "Train", k, n_screens, "Variable", "Coefficients", l)
                    append_results(results, epoch, "Train", k, n_screens, "Variable", "Reconstruction", r)
                    var_bias_c.append(bc)
                    var_bias_r.append(br)

                data = run_epoch(variable_trainer, 
                                 cfg.testing_size,
                                 testing_asterisms,
                                 atmosphere_test,
                                 cfg.basis,
                                 train=False,
                                 wfs=shwfs,
                                 tiptilt_removal=cfg.wavefront_tip_tilt_removal)
                
                for k, (l, r, _, _) in data.items():
                    append_results(results, epoch, "Test", k, cfg.testing_size, "Variable", "Coefficients", l)
                    append_results(results, epoch, "Test", k, cfg.testing_size, "Variable", "Reconstruction", r)

                if (epoch - 1) % cfg.save_frequency == 0 and not cfg.train_combined:
                    variable_trainer.save_models(epoch, suffix="variable", dir=f"{directory}/{cfg.start_time}")
                    variable_trainer.save_models(epoch, suffix="variable", dir=f"{directory}/current")

                save_asterism_data["variable"] += (epoch, training_asterisms + testing_asterisms)

            if (epoch - 1) % cfg.save_frequency == 0 and cfg.train_combined:
                trainer.save_models(epoch, dir=f"{directory}/{cfg.start_time}")
                trainer.save_models(epoch, dir=f"{directory}/current")
            
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            log.info(f"tracemalloc: {top_stats[:3]}")

    # Save final model
    if cfg.train_combined:
        trainer.save_models(dir=f"{directory}/{cfg.start_time}")
        trainer.save_models(dir=f"{directory}/current")
    else:
        fixed_trainer.save_models(suffix="fixed", dir=f"{directory}/{cfg.start_time}")
        fixed_trainer.save_models(suffix="fixed", dir=f"{directory}/current")
        variable_trainer.save_models(suffix="variable", dir=f"{directory}/{cfg.start_time}")
        variable_trainer.save_models(suffix="variable", dir=f"{directory}/current")

    # Run verification test
    log.info(f"Running final verification test")
    for runner in variable_trainer.runners:
        try:
            runner.fixed_only = False
        except AttributeError:
            pass

    verification_asterisms = []

    for i in range(cfg.testing_size):

        log.info(f"Running verification test screen {i+1}/{cfg.testing_size}")

        verification_atmosphere = phase_screen.Atmosphere(1, cfg.altitudes, cfg.r0, cfg.diameter, cfg.aperture_size)
        if not cfg.variable_off:
            if not cfg.train_combined:
                trainer = variable_trainer
            verification_asterism = asterisms.generate_variable_asterisms(1, cfg.magnitude, zero_rate=0.0, fixed_rate=0.0, mask=mask)
        else:
            if not cfg.train_combined:
                trainer = fixed_trainer
            verification_asterism = asterisms.generate_fixed_asterisms(1, cfg.fixed_asterism_angle, mask=mask, theta_0_random=cfg.fixed_theta_0_randomised)

        data = run_epoch(trainer, 
                        1,
                        verification_asterism,
                        verification_atmosphere,
                        cfg.basis,
                        train=False,
                        wfs=shwfs,
                        tiptilt_removal=cfg.wavefront_tip_tilt_removal)
        
        for k, (l, r, _, _) in data.items():
            append_results(results, epoch, "Verification", k, cfg.testing_size, "Variable", "Coefficients", l)
            append_results(results, epoch, "Verification", k, cfg.testing_size, "Variable", "Reconstruction", r)
        
        verification_asterisms.append(verification_asterism[0])

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
            with open(f"{dir}/results.csv", "w") as f:
                df.to_csv(f, sep=",")

        # 15/04/25 Removing these as I haven't really found any use for keeping them.
        # with open(f"{dir}/asterisms_variable.txt", "w") as f:
        #     for l in save_asterism_data["variable"]:
        #         f.write(f"{l}\n")

        # with open(f"{dir}/asterisms_fixed.txt", "w") as f:
        #     for l in save_asterism_data["fixed"]:
        #         f.write(f"{l}\n")

        # with open(f"{dir}/asterisms_verification.txt", "w") as f:
        #     for l in verification_asterisms:
        #         f.write(f"{l}\n")

        if not cfg.train_combined:
            trainer = fixed_trainer

        for k, m in trainer.models.items():
            name = m.__class__.__name__
            source = inspect.getsource(m.__class__)
            with open(f"{dir}/definitions/{name}.def", "w") as f:
                f.write(source)

    shutil.make_archive(f'{directory}/{cfg.start_time}', 'zip', f'{directory}/{cfg.start_time}')
    shutil.rmtree(f'{directory}/{cfg.start_time}')

    if len(fixed_bias_c) > 0:
        print(f"Bias (Fixed, Coefficient) = {np.mean(fixed_bias_c)}")
    if len(fixed_bias_r) > 0:
        print(f"Bias (Fixed, Reconstruction) = {np.mean(fixed_bias_r)}")
    if len(var_bias_c) > 0:
        print(f"Bias (Variable, Coefficient) = {np.mean(var_bias_c)}")
    if len(var_bias_r) > 0:
        print(f"Bias (Variable, Reconstruction) = {np.mean(var_bias_r)}")

    return df