from ml import ml
import log
from torch import nn

logger = log.get_logger(__name__)
log.setup_logger(logger)

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = self.__class__.__name__

        self.LRELU_grad = config.LReLU_grad
        self.do = config.dropout_rate
        self.n_coeffs = config.n_coeffs = 16

        logger.info(f"Initiating model {self.name}; n_modes={self.n_coeffs}")