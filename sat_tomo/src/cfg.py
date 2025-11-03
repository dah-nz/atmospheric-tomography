import argparse
import os

from torch import nn
from basis import basis
from datetime import datetime

class AltitudeExtendAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Get the number of layers from the layers argument
        n_layers = getattr(namespace, 'layers', None)
        
        # Validate the number of provided arguments
        if len(values) != n_layers:
            raise argparse.ArgumentTypeError(f"--altitudes: Expected {n_layers} altitude values, but got {len(values)}.")

        # Extend the list with the values
        existing_values = getattr(namespace, self.dest, [])
        if existing_values == None:
            existing_values = []
        existing_values.extend(map(float, values))
        setattr(namespace, self.dest, existing_values)

class Config():
    def __init__(self, time=None):

        if time is None:
            time = datetime.now()
        self.time = time
        self.start_time = time.strftime("%Y%m%d_%H%M%S")

        parser = argparse.ArgumentParser(prog='main.py',
                                     description="Runs different elements of the atmoshperic tomography simulator")
        
        parser.add_argument('-m', '--magnitude', action='store', default=14, type=int, help='Minimum limiting magnitude of background stars used for asterisms. All brighter stars will be used.')
        parser.add_argument('--max_magnitude', action='store', default=6, type=int, help='Maximum limiting magnitude of background stars used for finding asterisms. All brighter stars will be used.')
        parser.add_argument('-b', '--build_star_map', action='store_true', help='Build star maps')
        parser.add_argument('-s', '--satellites', action='store_true', help="Calculate satellite passes from fresh TLEs")
        parser.add_argument('-p', '--period', action='store', default=24.0, help="Period over which to calculate satellite passes (hours)")
        parser.add_argument('-t', '--train_models', action='extend', nargs="*", type=str, help='Train provided ML models')
        parser.add_argument('-e', '--epochs', action='store', default=64, type=int, help="Number of training epochs")
        parser.add_argument('-D', '--diameter', action='store', default=1.8, help="Set telescope diameter (m)")
        parser.add_argument('--directory', action='store', default='C:/Code/PhD/Data', help="Directory to save data to")
        parser.add_argument('--aperture_size', action='store', default=100, type=int, help="Set aperture size (pixels)")
        parser.add_argument('--r0', action='store', default=0.07, help="Set total Fried parameter across all layers (m)")
        parser.add_argument('--layers', action='store', default=3, type=int, help="Number of layers")
        parser.add_argument('--altitudes', action=AltitudeExtendAction, nargs="*", help="Layer altitudes")
        parser.add_argument('--training_size', action='store', default=256, type=int, help="Size of training batch")
        parser.add_argument('--testing_size', action='store', default=128, type=int, help="Size of test batch")
        parser.add_argument('--verification_size', action='store', default=-1, type=int, help="Size of verification batch. Defaults to same size as testing batch.")
        parser.add_argument('--fixed_asterism_angle', action='store', default=30.0, help="Angle for fixed asterism (arcseconds)")
        parser.add_argument('--basis_function', action='store', default="Zernike", type=str, help="Set of basis functions")
        parser.add_argument('-n', '--num_modes', action='store', default=21, type=int, help="Number of basis modes")
        parser.add_argument('--regime', action='store', default='default', type=str, help="Machine learning training regime")
        parser.add_argument('--LReLU_grad', action='store', default=0.05, help="Leaky Rectified Linear Unit (LReLU) leaky gradient")
        parser.add_argument('--dropout_rate', action='store', default=0.5, help="ML training dropout rate")
        parser.add_argument('--loss_function', action='store', default='mse_loss', help="ML training loss function")
        parser.add_argument('--learning_rate', action='store', default=1e-3, help="ML training AdamW learning rate")
        parser.add_argument('--weight_decay', action='store', default=1e-5, help="ML training AdamW weight decay")
        parser.add_argument('--save_frequency', action='store', default=16, type=int, help='Frequency to save intermediary model parameters')
        parser.add_argument('--limiting_angle', action='store', default=120, type=int, help='Limiting outer angle for variable asterisms')

        parser.add_argument('--num_sats', action='store', default=500, type=int, help="Number of satellites to calculate asterisms from")
        parser.add_argument('--sat_del_t', action='store', default=30, type=int, help="Time delta between satellite location checks (s)")
        parser.add_argument('--asterism_size', action='store', default=3, type=int, help="Number of stars in asterisms")
        parser.add_argument('--asterism_quality', action='store', default=0.0, type=float, help='Lower limit for selecting asterisms based on their predicted quality (Predicted Log of Relative Strehl)')
        parser.add_argument("--minimum_asterism_angle", action="store", default=0.0, type=float, help='Minimum distance for guide stars from the target')
        parser.add_argument("--shuffle_asterism", action='store_true', help="Shuffle each asterism before performing tomography")
        
        parser.add_argument('--shwfs', action='store_true', help="Use Shack-Hartmann WFS for estimating guide-star wavefront modes")
        parser.add_argument('--noise', action='store_true', help='Turn on noise in the SHWFS')

        parser.add_argument('--display_asterisms', action='store_true', help="Display data about asterisms")
        parser.add_argument('--display_training', action='store_true', help="Display results of training")

        parser.add_argument('--lsq_off', action='store_true', help='Turn off Least Squares tomography comparison')
        parser.add_argument('--glao_off', action='store_true', help='Turn off GLAO tomography comparison')
        parser.add_argument('--ao_off', action='store_true', help="Turn off adaptive optics comparison")

        parser.add_argument('--per_layer_r0', action='store_true', help="Set r0 value as each layer, rather than total across layers.")
        parser.add_argument('--fixed_theta_0_randomised', action="store_true", help="Randomise the starting angle for fixed asterisms (rotating the equally spaced guide-stars about the centre)")
        parser.add_argument('-l', '--load_previous', action='store_true', help="Load models from previous training run")
        parser.add_argument('-c', '--continue_training', action='store_true', help="Save results appended to previous training run")
        parser.add_argument("--mode_skip", action='store', default=3, help="Skip modes from the beginning. Default=3, skipping piston, tip/tilt.")
        parser.add_argument("--wavefront_tip_tilt_removal", action='store_true', help="Remove tip/tilt component from wavefronts prior to wavefront sensing, based on target tip/tilt value, approximating a tip/tilt mirror in the optical path")

        try:
            args = parser.parse_args()
        except argparse.ArgumentTypeError as e:
            print(e)
            quit()

        if type(args.sat_del_t) is str:
            args.sat_del_t = float(args.sat_del_t)

        if type(args.period) is str:
            args.period = float(args.period)

        if type(args.diameter) is str:
            args.diameter = float(args.diameter)

        try:
            if type(args.asterism_quality) is str:
                args.asterism_quality = float(args.asterism_quality)
        except ValueError:
             args.asterism_quality = None

        if type(args.r0) is str:
            args.r0 = float(args.r0)

        if type(args.fixed_asterism_angle) is str:
            args.fixed_asterism_angle = float(args.fixed_asterism_angle)

        if type(args.asterism_size) is str:
            args.asterism_size = int(args.asterism_size)

        if type(args.mode_skip) is str:
            args.mode_skip = int(args.mode_skip)

        if type(args.minimum_asterism_angle) is str:
            args.minimum_asterism_angle = float(args.minimum_asterism_angle)

        if args.verification_size <= 0:
            args.verification_size = args.testing_size

        if args.altitudes is None or len(args.altitudes) == 0:
            args.altitudes = [1000, 5000, 10000]

        if args.layers != len(args.altitudes):
            parser.error(f"Expected {args.layers} altitude values, but got {len(args.altitudes)}. --altitudes is required when --layers is set.")

        for k, v in vars(args).items():
            if not k.startswith("__"):
                setattr(self, k, v)


        #   Name                Run in parallel? , Fixed asterisms only?
        self.tomo_methods = {
            "LeastSquares":     {'run': True },
            "GLAO":             {'run': True },
            "AO":               {'run': True }
            }

        if self.lsq_off:
            self.tomo_methods['LeastSquares']['run'] = False
        if self.glao_off:
            self.tomo_methods["GLAO"]['run'] = False
        if self.ao_off:
            self.tomo_methods["AO"]['run'] = False
        
        self.basis = basis.select(self.basis_function, self.num_modes)
        
        max_screens = min(self.testing_size, self.training_size)

        # regime is [(num epochs, screens per epoch)]
        match self.regime:
            case 'screen_scaling':
                n = 8
                scaling_factor = 1.44
                self.regime = [(n, 2)]
                sum = self.regime[-1][0]
                while sum < self.epochs:
                    self.regime.append((self.regime[-1][0], min(max_screens, int(self.regime[-1][1] * scaling_factor + 0.5))))
                    sum += self.regime[-1][0]
            case 'linear_scaling':
                n = 8
                linear_factor = 10
                self.regime = [(n, 2)]
                sum = self.regime[-1][0]
                while sum < self.epochs:
                    self.regime.append((self.regime[-1][0], min(max_screens, int(self.regime[-1][1] + linear_factor))))
                    sum += self.regime[-1][0]
            case _: #default
                self.regime = [(self.epochs, max_screens)]

        match self.loss_function:
            case _: #mse_loss
                self.loss_function = nn.MSELoss()

        self.args = args

    def get_file_name(self, name):
        return f"{name}_{type(self.basis).__name__}_{self.num_modes}_{self.r0}_{self.fixed_asterism_angle}_{self.diameter}"
    
    def save(self, directory):

        with open(os.path.join(directory, "config.txt"), 'w') as f:

            for arg_name, arg_data in vars(self.args).items():
                f.write(f"{arg_name},{arg_data}\n")