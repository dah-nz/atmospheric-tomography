from torch import nn
from basis import basis
from datetime import datetime

class Config():
    def __init__(self, time=None):

        if time is None:
            time = datetime.now()
        self.start_time = time.strftime("%Y%m%d_%H%M%S")
        
        self.n_coeffs = 16
        self.basis_type = "zernike"
        self.basis = basis.select(self.basis_type, self.n_coeffs)

        self.D =                    1.8 # metres
        self.aperture_size =        64 # pixels
        self.layer_conditions =     [0.05, 0.05, 0.05] # metres
        self.altitudes =            [1000, 5000, 10000] # metres
        self.fixed_asterism_angle = 30 # arcseconds
        self.asterism_magnitude =   14 # M

        self.model_names = ['MLNN_baseline']
        self.asterism_types = ['fixed', 'variable']

        #   Name                Fixed asterisms only
        self.tomo_methods = {
            "LeastSquares":     True,
            "GLAO":             False,
            "Average":          False,
            "ScaledAverage":    False
            }

        TRAINING_SIZE = 128
        TEST_SIZE = 64
        #   num epochs, screens per epoch, training batch size, testing_batch_size
        self.regime = [
            ( 1,  4, TRAINING_SIZE, TEST_SIZE),
            ( 1,  8, TRAINING_SIZE, TEST_SIZE),
            ( 1, 16, TRAINING_SIZE, TEST_SIZE),
            ( 1, 32, TRAINING_SIZE, TEST_SIZE),
            ( 1, 64, TRAINING_SIZE, TEST_SIZE)]
            # ( 8,  4, TRAINING_SIZE, TEST_SIZE),
            # (12,  8, TRAINING_SIZE, TEST_SIZE),
            # (12, 16, TRAINING_SIZE, TEST_SIZE),
            # (16, 32, TRAINING_SIZE, TEST_SIZE),
            # (16, 64, TRAINING_SIZE, TEST_SIZE)]

        self.LReLU_grad = 0.01
        self.dropout_rate = 0.3
        self.loss_function = nn.MSELoss()

        self.learning_rate = 1e-3
        self.weight_decay = 1e-5

    def get_file_name(self, name):
        return f"{name}_{self.basis_type}_{self.n_coeffs}_{self.layer_conditions[0]}_{self.fixed_asterism_angle}_{self.D}"