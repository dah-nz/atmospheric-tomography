from ml import ml
import log
from torch import nn

logger = log.get_logger(__name__)
log.setup_logger(logger)

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = self.__class__.__name__

        self.learning_rate = config.learning_rate
        self.LRELU_grad = config.LReLU_grad
        self.weight_decay = config.weight_decay

        self.do = config.dropout_rate
        self.mode_skip = config.mode_skip

        self.n_features = config.num_modes - config.mode_skip
        self.asterism_features = 2 * config.asterism_size # theta, gamma and magnitude

        print(f"Initiating model {self.name}; n_features={self.n_features}")

        logger.info(f"Initiating model {self.name}; n_features={self.n_features}")