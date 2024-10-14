import cfg as config
import log as logging
import training

if __name__ == "__main__":
    cfg = config.Config()

    log = logging.get_logger(__name__)
    logging.setup_logger(log, start_time=cfg.start_time, to_console=True)

    log.info(" > Start model training < ")

    training.run(cfg)

    log.info(" > Model training complete <")
