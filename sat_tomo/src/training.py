from ml import ml, data
import log as logging

from tomography import asterisms, phase_screen

def run_epoch(trainer, n_train, n_test, asterism_train, asterism_test, atmosphere, basis):
    training_data = data.TomographyData(n_train, asterism_train, atmosphere, basis)
    test_data = data.TomographyData(n_test, asterism_test, atmosphere, basis)
    
    training_results = trainer.train(training_data)
    test_results = trainer.test(test_data)

    return (training_results, test_results)

def run(cfg):
    log = logging.get_logger(__name__)
    logging.setup_logger(log, start_time=cfg.start_time, to_console=True)
    
    fixed_trainer = ml.Trainer(cfg)
    variable_trainer = ml.Trainer(cfg)
    epoch = 0
    fixed_results = {}
    variable_results = {}
    total_epochs = sum([r[0] for r in cfg.regime])
    for n_epochs, n_screens, training_size, testing_size in cfg.regime:
        for e in range(n_epochs):
            log.info(f"Running Epoch {epoch}/{total_epochs} (Sub-epoch {e}/{n_epochs})")
            atmosphere = phase_screen.Atmosphere(n_screens, cfg.altitudes, cfg.layer_conditions, cfg.D, cfg.aperture_size)

            if 'fixed' in cfg.asterism_types:
                log.info(f"Running fixed asterisms [{epoch}/{total_epochs}]({e}/{n_epochs})")
                fixed_asterisms = asterisms.generate_fixed_asterisms(cfg.fixed_asterism_angle)
                fixed_results[epoch] = run_epoch(fixed_trainer, 
                                                 training_size, 
                                                 testing_size,
                                                 fixed_asterisms,
                                                 fixed_asterisms,
                                                 atmosphere,
                                                 cfg.basis)
                fixed_trainer.save_models(epoch, suffix="fixed")
            
            if 'variable' in cfg.asterism_types:
                log.info(f"Running variable asterisms [{epoch}/{total_epochs}]({e}/{n_epochs})")
                variable_asterisms = asterisms.generate_variable_asterisms(training_size + testing_size, cfg.asterism_magnitude)
                variable_results[epoch] = run_epoch(variable_trainer, 
                                                    training_size, 
                                                    testing_size,
                                                    variable_asterisms[:training_size],
                                                    variable_asterisms[-testing_size:],
                                                    atmosphere,
                                                    cfg.basis)
                variable_trainer.save_models(epoch, suffix="variable")

            epoch += 1

    with open("data/models/results/fixed_results.txt", "w") as f:
        for k, v in fixed_results.items():
            f.write(f"{k},{v}\n")
        
    with open("data/models/results/variable_results.txt", "w") as f:
        for k, v in variable_results.items():
            f.write(f"{k},{v}\n")