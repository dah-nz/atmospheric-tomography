# Standard imports
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
import time
import traceback
import joblib
import math
import ast

import pandas as pd
import seaborn as sns

from itertools import combinations

# Import sklearn models and utility functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, SVR
from sklearn.manifold import TSNE
import umap

# Import pytorch for ML
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# import modules for application specific tomography functions
from tomography import asterisms, beacon, phase_screen
from tomography.tomography import GLAO, LeastSquares, get_coeffs_for_asterism, AO
from basis import zernike
import util

def run_tests():

    # test_asterism_generation()
    # test_phase_variance()
    
    f = Filter()
    # f.run_asterism_comparison()
    f.filter_asterisms()
    # load_angle_data_test()

    plt.show()

    return # END run_tests


def load_angle_data_test():
    """
    Displays .pak file output, to check that the filtering is saving in the same format as is required by the rest of my software
    """
    with open(f'data/satellites/angle_data_mag6.pak', 'rb') as f:
        data = joblib.load(f)

    print(data[:5])

    with open(f'data/satellites/angle_data_filtered.pak', 'rb') as f:
        data = joblib.load(f)

    print(data[:5])

    return # END load_angle_data_test

class Filter():
    """
    Class of filtering programs
    Defines the main test functions
    """
    def __init__(self):
        # Size of test set
        self.n_asterisms = 8
        self.n_atm = 100

        # Ratio of different asterism types. These should sum to between 0.0 and 1.0 (the difference betwen 1.0 and the sum is the number of real asterisms to use)
        self.fixed_rate = 0.92
        self.uniform_rate = 0.04
        self.offset_rate = 0.03

        # Atmospheric and tomography simulation paramters
        self.N = 21
        self.Z = zernike.Zernike(self.N)
        self.D = 1.8
        self.r0 = 0.25
        self.ap_pix = 100
        self.alts = [1000, 5000, 10000]
        self.mask = util.circle(self.ap_pix)
        self.target = beacon.Beacon(0, 0, 10, aperture_mask=self.mask)

        # Different tomography methods
        self.glao = GLAO(self.Z, self.D, self.N, self.alts, self.ap_pix)
        self.ao = AO(self.Z, self.D, self.N, self.alts, self.ap_pix)
        self.lsq = LeastSquares(self.Z, self.D, self.N, self.alts, self.ap_pix, r0=self.r0)

        # Different data points to compute while running each asterism
        self.tomo_methods = ['glao', 'tiptilt', 'ao', 'lsq', 'lsq_p', 'ol', 'ideal']
        self.metrics = ['coeff_rmse', 'recon_rmse', 'strehl']
        self.statistics = ['mean', 'std', 'min', 'lq', 'med', 'uq', 'max', 'coeffs']

        # When running a filter, use the following:
        self.filter_method =    'glao'          # Tomography method to use for filtering
        self.filter_metric =    'recon_rmse'    # Metric to use for filtering
        self.filter_statistic = 'mean'          # Statistic to use for filtering
        self.filter_k =         0.62             # Ratio of open loop (OL) that consitutes "good" for classification
        self.filter_direction = False           # Direction of good vs bad: True if value higher than OL is classified as good
        self.filter_fraction =  0.2             # Fraction of total data to use for classifying data (set lower than 1.0 for larger sets of data if computation takes too long)

        # Model configuration
        self.train_test_ratio = 0.2
        self.svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

        self.star_label = 'abcdefghijklmnopqrstuvwxyz'

        max_perturb = 1 # arcseconds
        n_perturbs = 5 # number of perturbations in each direction
        x = np.linspace(-max_perturb, max_perturb, n_perturbs)
        y = np.linspace(-max_perturb, max_perturb, n_perturbs)
        self.perturbations = np.stack(np.meshgrid(x, y)).T.reshape(-1, 2)
        self.max_perturbation = np.amax(np.abs(self.perturbations))
        self.perturbations = [tuple(p) for p in self.perturbations]

        return # END Filter.__init__

    def worker(self, atm, aq, rq, l):
        try:
            tstart = time.time()
            worker_id = os.getpid()
            l.acquire()
            while not aq.empty():
                i, asterism_group = aq.get()
                l.release()

                t0 = time.time()
                star_ids = list(range(len(asterism_group)))

                group_data = {}

                for asterism_ids in combinations(star_ids, 3):

                    asterism_id = ''.join([self.star_label[id] for id in asterism_ids])
                    asterism = [asterism_group[id] for id in asterism_ids]

                    td = time.time()

                    print(f"[{worker_id:5}] Simulating asterism {asterism_id} (asterism group {i+1}/{self.n_asterisms}). Time since start: {td-tstart:.2f}s")

                    results = {
                        p: {t: {m: [] for m in self.metrics} for t in self.tomo_methods } for p in self.perturbations
                    }

                    coeff_data = {
                        p: {} for p in self.perturbations
                    }

                    stats = {
                        p: {t: {m: {} for m in self.metrics} for t in self.tomo_methods } for p in self.perturbations
                    }


                    for perturbation in self.perturbations:

                        p_ast = asterisms.perturb_asterism(asterism, perturbation)
                
                        for screens in atm:
                            target_coeffs = self.target.cutout_coeffs(self.Z, self.D, self.alts, screens, atm.metapupil_diameters)
                            target_screen = self.target.total_cutout(self.D, self.alts, screens, atm.metapupil_diameters)
                            ast_coeffs = get_coeffs_for_asterism(p_ast, self.Z, self.D, screens, self.alts, atm.metapupil_diameters)
                            coeff_data[perturbation]['target'] = target_coeffs
                            coeff_data[perturbation]['guidestars'] = ast_coeffs

                            for tomo in self.tomo_methods:
                                match tomo:
                                    case 'glao':
                                        coeffs = self.glao.run(p_ast, self.target, ast_coeffs)
                                    case 'tiptilt':
                                        coeffs = np.zeros_like(target_coeffs)
                                        coeffs[:3] = target_coeffs[:3]
                                    case 'ao':
                                        coeffs = self.ao.run(p_ast, self.target, ast_coeffs)
                                    case 'lsq':
                                        coeffs = self.lsq.run(p_ast, self.target, ast_coeffs)
                                    case 'lsq_p': # Use baseline projection matrices, but measurements from perturbation
                                        coeffs = self.lsq.run(asterism, self.target, ast_coeffs)
                                    case 'ideal':
                                        coeffs = target_coeffs
                                    case _:
                                        coeffs = np.zeros_like(target_coeffs)

                                for metric in self.metrics:
                                    match metric:
                                        case 'recon_rmse':
                                            recon = self.Z.compose(target_screen.shape, coeffs)
                                            result = util.rmse(recon, target_screen, mask=self.mask)
                                        case 'coeff_rmse':
                                            result = util.rmse(coeffs, target_coeffs)
                                        case 'strehl':
                                            recon = self.Z.compose(target_screen.shape, coeffs)
                                            # fig, axs = plt.subplots(1, 3)
                                            # fig.suptitle(tomo)
                                            r = recon - target_screen
                                            # axs[0].imshow(recon)
                                            # axs[1].imshow(target_screen)
                                            # axs[2].imshow(r)
                                            var = np.var(r, where=self.mask > 0)
                                            # print(f"{tomo}\ntarget r0: {phase_screen.calculate_r0(target_screen, self.D)}\nrecon r0: {phase_screen.calculate_r0(recon, self.D)}\ndiff r0: {phase_screen.calculate_r0(r, self.D)}\nVariance: {var}\nStrehl: {np.exp(-var)}")
                                            result = np.exp(-var)
                                            # plt.show()
                                        case _:
                                            result = np.nan
                                    results[perturbation][tomo][metric].append(result)

                        for tomo in self.tomo_methods:
                            for metric in self.metrics:
                                for statistic in self.statistics:
                                    match statistic:
                                        case 'std':
                                            r = np.std(results[perturbation][tomo][metric])
                                        case 'mean':
                                            r = np.mean(results[perturbation][tomo][metric])
                                        case 'min':
                                            r = np.amin(results[perturbation][tomo][metric])
                                        case 'lq':
                                            r = np.percentile(results[perturbation][tomo][metric], 25)
                                        case 'med':
                                            r = np.median(results[perturbation][tomo][metric])
                                        case 'uq':
                                            r = np.percentile(results[perturbation][tomo][metric], 75)
                                        case 'max':
                                            r = np.amax(results[perturbation][tomo][metric])
                                        case 'raw':
                                            r = results[perturbation][tomo][metric]

                                    stats[perturbation][tomo][metric][statistic] = r
                    
                    group_data[asterism_id] = (stats, coeff_data)

                rq.put((i, group_data))

                t1 = time.time()
                dt = t1 - t0
                dt_total = t1 - tstart 
                print(f"[{worker_id:5}] Simulated asterism {i+1}/{self.n_asterisms}. Took {dt:5.2f}s. Total time since start: {dt_total:.2f}s")
                l.acquire()
            l.release()
        except Exception as e:
            print(type(e))    # the exception type
            print(e.args)     # arguments stored in .args
            print(e)
            print(traceback.format_exc(), flush=True)

        return # END Filter.worker

    def run_asterism_comparison(self):
        """
        Filter.run_asterism_comparison()

        Run a set of asterisms on a set of static phase screens.
        """
        all_asterisms = asterisms.get_test_asterisms_a_choose_b(self.n_asterisms)

        print(f"Generating {self.n_atm} simulated atmospheres")
        limit = asterisms.LIMITING_ANGLE + self.max_perturbation     # Increase limiting angle for atmosphere generation to account for asterism perturbations.
        atm = phase_screen.Atmosphere(self.n_atm, self.alts, r0=self.r0, D=self.D, aperture_size=self.ap_pix, limiting_angle=limit)
        print(f"Completed simulating {self.n_atm} atmospheres")

        manager = Manager()
        asterism_queue = manager.Queue()
        result_queue = manager.Queue()
        lock = manager.Lock()

        for i, asterism_group in tqdm(enumerate(all_asterisms)):
            asterism_queue.put((i, asterism_group))

        n_processes = 8
        print("Run multi-processing for asterism quality filtering")

        with Pool(processes=n_processes) as p:
            workers = [p.apply_async(self.worker, args=(atm, asterism_queue, result_queue, lock)) for _ in range(n_processes)]

            res = [p.get() for p in workers]

        print("Multi-processing completed")

        results = []

        while not result_queue.empty():
            j, group_data = result_queue.get()

            if j % 100 == 0:
                print(f"Loading results for asterism {j+1}/{self.n_asterisms}")
            asterism = all_asterisms[j]

            for ast_id in group_data.keys():
                for p in self.perturbations:
                    r_dict = {
                        'group_id':  j
                    }
                    for i in range(len(asterism)):
                        r_dict[f'gs_{self.star_label[i]}'] = asterism[i].to_simple_str()

                    r_dict['asterism_id'] = ast_id
                    r_dict['perturbation'] = p

                    for s in self.statistics:
                        for t in self.tomo_methods:
                            for m in self.metrics:
                                r_dict[f'{t}_{m}_{s}'] = group_data[ast_id][0][p][t][m][s]
                    r_dict['target_coeffs'] = repr(group_data[ast_id][1][p]['target'])
                    r_dict['asterism_coeffs'] = repr(group_data[ast_id][1][p]['guidestars'])

                    results.append(r_dict)

        df = pd.DataFrame(results)
        df.to_csv('data/satellites/asterism_quality_data.csv')

        def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
            circle = [util.pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
            X = [x+centre[0] for x, _ in circle]
            Y = [y+centre[1] for _, y in circle]
            return X, Y
    
        xcirc, ycirc = getCircle(120, 64)
    
        fig, axs = plt.subplots(1, 1)
        fig.suptitle("Asterisms")

        def plot_asterisms(ax, title, asts):
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.set_xlabel('Arcseconds')
            ax.set_ylabel('Arcseconds')
            for ast in asts:
                x, y = asterisms.get_asterism_coords(ast, theta_offset=True)
                ax.scatter(x, y, c='r', alpha=0.2)
            for ast in asts:
                x, y = asterisms.get_asterism_coords(ast)
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                ax.plot(x, y, 'b', alpha=0.1)
            ax.plot(xcirc, ycirc, 'k:', alpha=0.2)

        plot_asterisms(axs, "All Asterisms", all_asterisms)

        return # END Filter.run_asterism_comparison

    def filter_asterisms(self):
        """
        Filter asterisms for good and bad
        """
        # load configuration values
        tomo_method = self.filter_method
        metric =      self.filter_metric
        statistic =   self.filter_statistic
        k =           self.filter_k
        gt =          self.filter_direction     # True if higher value is better
        frac =        self.filter_fraction

        # Build dataframe from asterism quality data
        df = pd.read_csv('data/satellites/asterism_quality_data.csv') #, index_col="group_id")
        

        # df_no_perturb = df[df['perturbation'] == "(0.0, 0.0)"]
        # print(df_no_perturb)
        # print(df_no_perturb.shape)
        # Load baseline asterism data from initial dataframe
        # Angles and distances from the target

        asterism_list = []
        for idx in df['group_id'].unique():
            for id in df['asterism_id'].unique():
                # id = ''.join([self.star_label[i] for i in comb])
                # print(idx, id)
                asterism_list.append((idx, id, [beacon.Beacon.from_simple_str(df.loc[(df['group_id'] == idx) & (df['asterism_id'] == id), f'gs_{a}'].iloc[0]) for a in id]))

        # Build dataframe of features based on asterism geometry
        df0 = pd.DataFrame()

        df0['group_id'] = [group_id for group_id, _, _ in asterism_list]
        df0['asterism_id'] = [ast_id for _, ast_id, _ in asterism_list]
        df0['performance'] = [df.loc[(df['group_id'] == group_id)
                                      & (df['asterism_id'] == ast_id)
                                      & (df['perturbation'] == "(0.0, 0.0)"),
                                      f'{tomo_method}_{metric}_{statistic}'] for group_id, ast_id, _ in asterism_list]

        df0['distance_0'] = norm([a[0]._gamma_arcsec for _, _, a in asterism_list])
        df0['distance_1'] = norm([a[1]._gamma_arcsec for _, _, a in asterism_list])
        df0['distance_2'] = norm([a[2]._gamma_arcsec for _, _, a in asterism_list])
        df0['angle_0'] =    norm([a[0].theta for _, _, a in asterism_list])
        df0['angle_1'] =    norm([a[1].theta for _, _, a in asterism_list])
        df0['angle_2'] =    norm([a[2].theta for _, _, a in asterism_list])

        df0['ave_distance'] =          norm([asterisms.ave_distance(a) for _, _, a in asterism_list])
        df0['min_distance'] =          norm([asterisms.min_distance(a) for _, _, a in asterism_list])
        df0['max_distance'] =          norm([asterisms.max_distance(a) for _, _, a in asterism_list])
        df0['area'] =                  norm([asterisms.area(a) for _, _, a in asterism_list])
        df0['centroid_distance'] =     norm([asterisms.centroid_distance(a) for _, _, a in asterism_list])
        df0['circumcentre_distance'] = norm([asterisms.circumcentre_distance(a) for _, _, a in asterism_list])
        df0['circumcircle_radius'] =   norm([asterisms.circumcircle_radius(a) for _, _, a in asterism_list])
        df0['edge_length_ratio'] =     norm([asterisms.length_ratio(a) for _, _, a in asterism_list])
        df0['ave_edge_length'] =       norm([asterisms.average_side_length(a) for _, _, a in asterism_list])
        df0['min_edge_length'] =       norm([asterisms.minimum_side_length(a) for _, _, a in asterism_list])
        df0['max_edge_length'] =       norm([asterisms.maximum_side_length(a) for _, _, a in asterism_list])
        df0['incentre_distance'] =     norm([asterisms.incentre_distance(a) for _, _, a in asterism_list])
        df0['incircle_radius'] =       norm([asterisms.incircle_radius(a) for _, _, a in asterism_list])
        df0['perimeter_area_ratio'] =  norm([asterisms.perimeter_area_ratio(a) for _, _, a in asterism_list])
        df0['edge_length_extreme_stars'] = norm([asterisms.closest_furthest_side_length(a) for _, _, a in asterism_list])
        df0['distance_to_closest_edge'] =  norm([asterisms.distance_to_closest_line(a) for _, _, a in asterism_list])
        df0['perimeter'] =          norm([asterisms.perimeter(a) for _, _, a in asterism_list])
        df0['min_interior_angle'] = norm([asterisms.min_angle(a) for _, _, a in asterism_list])
        df0['max_interior_angle'] = norm([asterisms.max_angle(a) for _, _, a in asterism_list])
        df0['gaussian_area'] =      norm([asterisms.gaussian_area_within_triangle(a) for _, _, a in asterism_list])
        df0['aperture_overlap'] =   norm([asterisms.overlap(a, [1000, 5000, 10000]) for _, _, a in asterism_list])
        df0['target_in_triangle'] = norm([np.array(asterisms.is_target_in_triangle(a), dtype=np.float32) for _, _, a in asterism_list])

        # if frac < 1.0 and frac > 0.0:
        #     df0 = df0.sample(frac=frac)


        named_columns = [c for c in df0.columns if not c.startswith("Unnamed")]
        self.n_features = len(named_columns) - 3
        print(f"Number of features: {self.n_features}")

        # self.plot_features(df0)
        # self.comparison(df0)

        self.perturbation_analysis(df)

        # self.dimensionality_reduction(df0)
        # self.dimensionality_reduction_ML(df0)

        # self.data_correlation(df0)

        # self.run_classification_models(df0)
        # self.run_ML_classification(df0)

        return # END Filter.filter_asterisms
    
    def perturbation_analysis(self, df):
        tomo_methods = ['glao', 'lsq', 'lsq_p']
        metrics = ['coeff_rmse', 'recon_rmse']
        statistics = ['mean']

        plots = [(method, metric, stat) for method in tomo_methods for metric in metrics for stat in statistics]

        # errors = {}

        for method, metric, statistic in plots:
            print(f'Perturbation of {method}_{metric}_{statistic}')
            # errors[f'{method}_{metric}_{statistic}'] = {}
            baseline = df[df['perturbation'] == "(0.0, 0.0)"][['group_id', 'asterism_id', f'{method}_{metric}_{statistic}']]
            baseline = baseline.rename(columns={f'{method}_{metric}_{statistic}': f'{method}_{metric}_{statistic}_baseline'})

            df = df.merge(baseline, on=['group_id', 'asterism_id'], how='left')
            df[f'{method}_{metric}_{statistic}_diff'] = df[f'{method}_{metric}_{statistic}'] - df[f'{method}_{metric}_{statistic}_baseline']

            # for perturbation in df['perturbation'].unique():
            #     if perturbation != "(0.0, 0.0)":
            #         errors[f'{method}_{metric}_{statistic}'][perturbation] = []
            #         for group_id in df['group_id'].unique():
            #             for asterism_id in df['asterism_id'].unique():
            #                 perf0 = df.loc[(df['group_id'] == group_id) & (df['asterism_id'] == asterism_id) & (df['perturbation'] == "(0.0, 0.0)")]
            #                 perf = df.loc[(df['group_id'] == group_id) & (df['asterism_id'] == asterism_id) & (df['perturbation'] == perturbation)]
            #                 error = perf0[f"{method}_{metric}_{statistic}"].iloc[0] - perf[f"{method}_{metric}_{statistic}"].iloc[0]
            #                 errors[f'{method}_{metric}_{statistic}'][perturbation].append(error)

        for method, metric, statistic in plots:  
            fig, axs = plt.subplots(5, 5)
            fig.suptitle(f'Perturbation of {method}_{metric}_{statistic}')
            
            # Column labels
            for j, label in enumerate(['-1.0"', '-0.5"', '0.0"', '0.5"', '1.0"']):
                fig.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)

            # Row labels
            for i, label in enumerate(['1.0"', '0.5"', '0.0"', '-0.5"', '-1.0"']):
                fig.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)

            for perturbation in df['perturbation'].unique():
                if perturbation != "(0.0, 0.0)":
                    pert = eval(perturbation)

                    errors = df[df['perturbation'] == perturbation][f"{method}_{metric}_{statistic}_diff"].tolist()

                    c = int(pert[0] * 2. + 2.)
                    r = int(2. - pert[1] * 2.)
                    axs[r, c].hist(errors, bins=30, color='b')
                    # axs[r, c].set_xlabel('ΔRMSE (rads)')
                else:
                    values = df[df['perturbation'] == perturbation][f"{method}_{metric}_{statistic}"].tolist()
                    axs[2, 2].hist(values, bins=30, color='r')
                    # axs[2, 2].set_xlabel('RMSE (rads)')
                
            for i in range(5):
                axs[4, i].set_xlabel('RMSE (rads)')
                axs[i, 0].set_ylabel('Count')
                
            # fig.tight_layout()
        
        fig, axs = plt.subplots(5, 5)
        fig.suptitle("Perturbation measurement error")

        # Column labels
        for j, label in enumerate(['-1.0"', '-0.5"', '0.0"', '0.5"', '1.0"']):
            fig.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)

        # Row labels
        for i, label in enumerate(['1.0"', '0.5"', '0.0"', '-0.5"', '-1.0"']):
            fig.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)

        def to_arr(s):
            array = np.array
            float32 = np.float32
            return eval(s)

            # import re
            # s=re.sub('\[ +', '[', s.strip())
            # # Replace commas and spaces
            # s=re.sub('[,\s]+', ', ', s)
            # return np.array(ast.literal_eval(s))

        baseline_coeffs = df[df['perturbation'] == "(0.0, 0.0)"]["asterism_coeffs"].tolist()
        baseline_coeffs = np.array([to_arr(b) for b in baseline_coeffs])
        baseline_coeffs = baseline_coeffs.reshape(baseline_coeffs.shape[0], -1)

        for perturbation in df['perturbation'].unique():
            if perturbation != "(0.0, 0.0)":
                pert = eval(perturbation)

                coeffs = df[df['perturbation'] == perturbation]["asterism_coeffs"].tolist()
                coeffs = np.array([to_arr(b) for b in coeffs])
                coeffs = coeffs.reshape(coeffs.shape[0], -1)

                errors = [util.rmse(a, b) for a, b in zip(coeffs, baseline_coeffs)]

                c = int(pert[0] * 2. + 2.)
                r = int(2. - pert[1] * 2.)
                axs[r, c].hist(errors, bins=30, color='b')
            else:
                coeffs = df[df['perturbation'] == perturbation]["asterism_coeffs"].tolist()
                coeffs = np.array([to_arr(b) for b in coeffs])
                print(coeffs.shape)
                axs[2, 2].hist(np.sqrt(np.mean(coeffs*coeffs, axis=(1, 2))), bins=30, color='r')

        for i in range(5):
            axs[4, i].set_xlabel('ΔRMSE (rads)')
            axs[i, 0].set_ylabel('Count')

    
    def comparison(self, df):
        features = df.drop(columns=['performance', 'asterism_id', 'group_id'])
        feature_list = features.columns
        group_ids = df['group_id'].unique().tolist()
        asterism_ids = df['asterism_id'].unique().tolist()

        group_asterism_ids = [(gid, aid) for gid in group_ids for aid in asterism_ids]

        total = math.comb(len(group_asterism_ids), 2)

        pairs = combinations(group_asterism_ids, 2)
        asterism_pairs = combinations(asterism_ids, 2)

        data = {}
        for feature in feature_list:
            data[feature] = { 'greater': 0, 'less than': 0, 'equal': 0, 'equal performance': 0 }

        total /= 100
        eps = 1e-3

        rng = np.random.default_rng()
        pairs = rng.choice(pairs, total)

        print(len(group_asterism_ids), total)

        for ga0, ga1 in tqdm(pairs, total=total):
            g0, a0 = ga0
            g1, a1 = ga1

            p0 = df.loc[(df['group_id'] == g0) & (df['asterism_id'] == a0), 'performance'].values[0]
            p1 = df.loc[(df['group_id'] == g1) & (df['asterism_id'] == a1), 'performance'].values[0]
            for feature in feature_list:
                v0 = df.loc[(df['group_id'] == g0) & (df['asterism_id'] == a0), feature].values[0]
                v1 = df.loc[(df['group_id'] == g1) & (df['asterism_id'] == a1), feature].values[0]

                if v1 == v0:
                    data[feature]['equal'] += 1
                elif p1 < (p0 + eps) and p1 > (p0 - eps):
                    data[feature]['equal performance'] += 1
                elif p0 < p1:
                    data[feature]['greater'] += int(v0 > v1)
                    data[feature]['less than'] += int(v0 < v1)
                elif p0 > p1: # p0 performed worse
                    data[feature]['greater'] += int(v1 > v0)
                    data[feature]['less than'] += int(v1 < v0)
            
            # total += 1
            

        # for group_id in group_ids:
        #     for a0, a1 in asterism_pairs:
        #         p0 = df.loc[(df['group_id'] == group_id) & (df['asterism_id'] == a0), 'performance'].values[0]
                
        #         p1 = df.loc[(df['group_id'] == group_id) & (df['asterism_id'] == a1), 'performance'].values[0]
        #         for feature in feature_list:
        #             v0 = df.loc[(df['group_id'] == group_id) & (df['asterism_id'] == a0), feature].values[0]
        #             v1 = df.loc[(df['group_id'] == group_id) & (df['asterism_id'] == a1), feature].values[0]
                    

        #             if v1 == v0:
        #                 data[feature]['equal'] += 1
        #             elif p1 < (p0 + eps) and p1 > (p0 - eps):
        #                 data[feature]['equal performance'] += 1
        #             elif p0 < p1:
        #                 data[feature]['greater'] += int(v0 > v1)
        #                 data[feature]['less than'] += int(v0 < v1)
        #             elif p0 > p1: # p0 performed worse
        #                 data[feature]['greater'] += int(v1 > v0)
        #                 data[feature]['less than'] += int(v1 < v0)

        #         total += 1

        # X locations for the groups
        x = np.arange(len(feature_list))  # [0, 1, 2]
        width = 0.7 / 4  # width of each bar

        gt = [data[f]['greater'] for f in feature_list]
        lt = [data[f]['less than'] for f in feature_list]
        eq = [data[f]['equal'] for f in feature_list]
        ep = [data[f]['equal performance'] for f in feature_list]

        lt = np.array(lt) / total
        gt = np.array(gt) / total
        eq = np.array(eq) / total
        ep = np.array(ep) / total

        # Plot grouped bars
        plt.plot([-width, max(x)+width], [0.5, 0.5], 'k:', alpha=0.2)
        plt.bar(x - 3*width/2, gt, width, label='Better performance when larger')
        plt.bar(x - width/2, lt, width, label='Better performance when smaller')
        plt.bar(x + width/2, eq, width, label='Values equal')
        plt.bar(x + 3*width/2, ep, width, label='Performance equal')

        # Add labels, title, ticks, and legend
        plt.xlabel('Feature')
        plt.ylabel('Value')
        plt.title('Relative Performance')
        plt.xticks(x, feature_list, rotation=90)
        plt.legend()
        plt.tight_layout()

    
    def plot_features(self, df):
        rows = 2
        cols = 3

        figi = 0
        fig, axs = plt.subplots(rows, cols)
        fig.suptitle(f"Features {figi+1}")

        X = df.drop(columns=['performance', 'asterism_id', 'group_id'])
        y = df['performance']

        for i, col in enumerate(X.columns):
            if i >= (figi+1) * (rows * cols):
                fig.tight_layout()
                fig, axs = plt.subplots(rows, cols)
                fig.suptitle(f"Features {figi+1}")
                figi += 1
            r = (i - (figi * rows * cols)) // cols
            c = (i - (figi * rows * cols)) % cols
            axs[r, c].scatter(df[col].to_list(), y)
            axs[r, c].set_title(col)
        
        fig.tight_layout()

    def dimensionality_reduction(self, df):
        X = df.drop(columns=['performance', 'asterism_id', 'group_id'])
        y = df['performance']

        print("Calculating 2D UMAP")
        plt.figure()
        umap_reducer = umap.UMAP(n_components=2)
        X_umap = umap_reducer.fit_transform(X)
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
        plt.title("UMAP 2D")

        # print("Calculating 3D UMAP")
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # umap_reducer = umap.UMAP(n_components=3)
        # X_umap = umap_reducer.fit_transform(X)
        # ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=y, cmap='viridis')
        # fig.suptitle("UMAP 3D")

        # print("Calculating 2D t-SNE")
        # tsne = TSNE(n_components=2)
        # X_tsne = tsne.fit_transform(X)

        # plt.figure()
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
        # plt.title("t-SNE 2D")

        # print("Calculating 3D t-SNE")
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # tsne2 = TSNE(n_components=3)
        # X_tsne2 = tsne2.fit_transform(X)
        # ax.scatter(X_tsne2[:, 0], X_tsne2[:, 1], X_tsne2[:, 2], c=y, cmap='viridis')
        # fig.suptitle("t-SNE 3D")

        # # Display higher dimensional data as 2D using Principal Component Analysis
        # print("Calculating 2D PCA")
        # pca = PCA(n_components=2)
        # X_pca = pca.fit_transform(X)

        # plt.figure()
        # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
        # plt.title("PCA 2D")

        # print("Calculating 3D PCA")
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # PCA_model = PCA(n_components=3).fit(X)
        # X_reduced = PCA_model.transform(X)
        # ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis')
        # fig.suptitle("PCA 3D")

        n_components = 10
        if n_components < self.n_features:
            # Display higher dimensional data as a PCA -> t-SNE analysis
            print("Calculating PCA->t-SNE 2D analysis")
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)

            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(X_pca)

            plt.figure()
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
            plt.title("PCA -> t-SNE 2D")

            print("Calculating PCA->UMAP 2D analysis")
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)

            tsne = umap.UMAP(n_components=2)
            X_tsne = tsne.fit_transform(X_pca)

            plt.figure()
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
            plt.title("PCA -> UMAP 2D")

        return # END Filter.dimensionality_reduction
        
    def data_correlation(self, df):
        # Show correlations between variables
        plt.figure()
        correlations = df.corr()
        sns.heatmap(correlations, annot=True)

        return # END Filter.data_correlation

    def classification_function(self, performance, df_raw):
        if self.filter_direction is True:
            y = (performance > self.filter_k * (df_raw[f'ol_{self.filter_metric}_{self.filter_statistic}'].mean())).astype(int)
        else:
            y = (performance < self.filter_k * (df_raw[f'ol_{self.filter_metric}_{self.filter_statistic}'].mean())).astype(int)

        return y
        # END Filter.classification_data

    def split_data(self, df, classification=None):
        X = df.drop(columns=['performance', 'asterism_id', 'group_id'])

        if classification:
            y = df['class']
        else:
            y = df['performance']

        return train_test_split(X, y, test_size=self.train_test_ratio)
        #END Filter.split_data

    def run_regression_models(self, df):
        # Data for regession testing
        X_trn, X_tst, y_trn, y_tst = self.split_data(df)

        # Partial Least Squares Regression (PLS)
        model = PLSRegression(n_components=2)
        regression(X_trn, X_tst, y_trn, y_tst, model, "Partial Least Squares Regression")

        # Principal Component Regression (PCR)
        pcr = make_pipeline(PCA(n_components=2), LinearRegression())
        regression(X_trn, X_tst, y_trn, y_tst, pcr, "Principal Component Regression")

        # Ridge Regression
        rdg = Ridge(alpha=0.1)
        regression(X_trn, X_tst, y_trn, y_tst, rdg, "Ridge Regression")
        
        # Lasso Regression
        lso = Lasso(alpha=0.1)
        regression(X_trn, X_tst, y_trn, y_tst, lso, "Lasso Regression")

        for kernel in self.svm_kernels:
            svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
            regression(X_trn, X_tst, y_trn, y_tst, svr, f"Support Vector Regression [{kernel}]")

        return # END Filter.run_regression_models


    def run_classification_models(self, df):
        # Data for classification testing
        X_trn, X_tst, y_trn, y_tst = self.split_data(df, classification=True)
        # Linear Discriminant Analysis
        lda = LinearDiscriminantAnalysis()
        classification(X_trn, X_tst, y_trn, y_tst, lda, "Linear Discriminant Analysis")

        # Support Vector Machine
        # for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for kernel in self.svm_kernels:
            clf = SVC(kernel=kernel)
            classification(X_trn, X_tst, y_trn, y_tst, clf, f"Support Vector Classifier [{kernel}]")

        return # END Filter.run_regression_models

    def run_ML_regression(self, df):

        X_trn, X_tst, y_trn, y_tst = self.split_data(df)

        n_features = self.n_features

        class RegressionModel(nn.Module):
            def __init__(self):
                super(RegressionModel, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(n_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            def forward(self, x):
                return self.network(x)
 
        reg_model = ML_Model(RegressionModel, nn.MSELoss)
        regression(X_trn, X_tst, y_trn, y_tst, reg_model, "ML Regression")

        return # END Filter.run_ML_regression
    
    def run_ML_classification(self, df):
        X_trn, X_tst, y_trn, y_tst = self.split_data(df, classification=True)

        n_features = self.n_features

        class ClassificationModel(nn.Module):
            def __init__(self):
                super(ClassificationModel, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(n_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)
                )
            def forward(self, x):
                return self.network(x)

        cls_model = ML_Model(ClassificationModel, nn.CrossEntropyLoss, cls=True)
        classification(X_trn, X_tst, y_trn, y_tst, cls_model, "ML Classification")

    def dimensionality_reduction_ML(self, df):
        X_trn, X_tst, y_trn, y_tst = self.split_data(df)

        n_features = self.n_features

        class Autoencoder(nn.Module):
            def __init__(self):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                )

                self.decoder = nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, n_features)
                )
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
           
        print("Calculating ML -> UMAP 2D (Auto-encoder)")
        ae_model = ML_AE(Autoencoder, nn.MSELoss, cls=False)

        ae_model.fit(X_trn, X_trn)

        Z_encoding = ae_model.encode(X_tst)

        tsne = umap.UMAP(n_components=2)
        X_tsne = tsne.fit_transform(Z_encoding)

        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tst, cmap='viridis')
        plt.title("Autoencoder -> UMAP 2D")

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(Z_encoding)

        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tst, cmap='viridis')
        plt.title("Autoencoder -> TSNE 2D")

        class Autodecoder(nn.Module):
            def __init__(self):
                super(Autodecoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, 32),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(64, 128),
                )

                self.decoder = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(32, n_features)
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        print("Calculating ML -> UMAP 2D (Auto-decoder)")
        ae_model = ML_AE(Autodecoder, nn.MSELoss, cls=False)

        ae_model.fit(X_trn, X_trn)

        Z_encoding = ae_model.encode(X_tst)

        tsne = umap.UMAP(n_components=2)
        X_tsne = tsne.fit_transform(Z_encoding)

        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tst, cmap='viridis')
        plt.title("Autodecoder -> UMAP 2D")

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(Z_encoding)

        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tst, cmap='viridis')
        plt.title("Autodecoder -> TSNE 2D")

        return # END Filter.dimensionality_reduction_ML
    
    # END_CLASS Filter

def test_asterism_generation():

    fixed_asts =   asterisms.get_test_asterisms(100, fixed_rate=1.0, uniform_rate=0.0, offset_rate=0.0)
    uniform_asts = asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=1.0, offset_rate=0.0)
    offset_asts =  asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=0.0, offset_rate=1.0)
    real_asts =    asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=0.0, offset_rate=0.0)
    mixed_asts =   asterisms.get_test_asterisms(100, fixed_rate=0.1, uniform_rate=0.2, offset_rate=0.5)
    mix_2_asts =   asterisms.get_test_asterisms(100, fixed_rate=0.2, uniform_rate=0.2, offset_rate=0.4)
    

    def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
        circle = [util.pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
        X = [x+centre[0] for x, _ in circle]
        Y = [y+centre[1] for _, y in circle]
        return X, Y
    
    xcirc, ycirc = getCircle(120, 64)
    
    fig, axs = plt.subplots(2, 3)
    fig.suptitle("Asterism Distribution")

    def plot_asterisms(ax, title, asts):
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel('Arcseconds')
        ax.set_ylabel('Arcseconds')
        for ast in asts:
            x, y = asterisms.get_asterism_coords(ast, theta_offset=True)
            ax.scatter(x, y, c='r', alpha=0.2)
        for ast in asts:
            x, y = asterisms.get_asterism_coords(ast)
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax.plot(x, y, 'b', alpha=0.1)
        ax.plot(xcirc, ycirc, 'k:', alpha=0.2)

    plot_asterisms(axs[0, 0], "Fixed Asterisms", fixed_asts)
    plot_asterisms(axs[0, 1], "Uniform Asterisms", uniform_asts)
    plot_asterisms(axs[1, 0], "Offset Asterisms", offset_asts)
    plot_asterisms(axs[1, 1], "Real Asterisms", real_asts)
    plot_asterisms(axs[0, 2], "Mix 1 Asterisms", mixed_asts)
    plot_asterisms(axs[1, 2], "Mix 2 Asterisms", mix_2_asts)

    fig.tight_layout()

    return # END test_asterism_generation

def test_phase_variance():
    cutout = 100
    phase = 120
    n_modes = 21
    Z = zernike.Zernike(n_modes)

    for i in range(1, n_modes):
        c = np.array([0]*n_modes)
        c[i] = 1
        s = Z.compose(cutout, c)
        # print(np.var(s, where=util.circle(cutout) > 0))

        s = Z.compose(phase, c)
        p_var = np.var(s, where=util.circle(phase) > 0)
        # print(p_var)

        s2 = Z.compose(phase, c)
        c0 = (phase - cutout) // 2
        c1 = (phase + cutout) // 2
        c_var = np.var(s2[c0:c1, c0:c1], where=util.circle(cutout) > 0)
        # print(c_var)

        k = np.sqrt(p_var / c_var)
        s2 = s2 * k

        c_var2 = np.var(s2[c0:c1, c0:c1], where=util.circle(cutout) > 0)
        # print(c_var2)

        print(f"Mode: {i+1}\n\tPhase Size: {phase}\n\tCutout Size: {cutout}\n\tPhase Var: {p_var}\n\tCutout Var: {c_var}\n\tScaled Var: {c_var2}\n\tScaling Factor: {k}")

    c = np.ones_like(c) * np.sqrt(1/20)
    s = Z.compose(phase, c)
    p_var = np.var(s, where=util.circle(phase) > 0)
    c0 = (phase - cutout) // 2
    c1 = (phase + cutout) // 2
    c_var = np.var(s[c0:c1, c0:c1], where=util.circle(cutout) > 0)
    k = np.sqrt(p_var / c_var)
    s2 = s * k
    c_var2 = np.var(s2[c0:c1, c0:c1], where=util.circle(cutout) > 0)

    print(f"Modes: {n_modes}\n\tPhase Size: {phase}\n\tCutout Size: {cutout}\n\tPhase Var: {p_var}\n\tCutout Var: {c_var}\n\tScaled Var: {c_var2}\n\tScaling Factor: {k}")

    k_data = []
    calc_data = []
    for i in range(100):
        ks = []
        p_vars = []
        c_vars = []
        scaled_vars = []
        p_gen = phase_screen.PhaseScreenGenerator(phase, 1.8, 0.07)
        screens = p_gen.generate(n_modes)

        for screen in screens:
            # screen = phase_screen.kolmogorov((phase, phase), 1.8, 0.07)
            p_var = np.var(screen, where=util.circle(phase) > 0)
            p_vars.append(p_var)
            c0 = (phase - cutout) // 2
            c1 = (phase + cutout) // 2
            c_var = np.var(screen[c0:c1, c0:c1], where=util.circle(cutout) > 0)
            c_vars.append(c_var)
            k = np.sqrt(p_var / c_var)
            s2 = screen * k
            c_var2 = np.var(s2[c0:c1, c0:c1], where=util.circle(cutout) > 0)
            scaled_vars.append(c_var2)
            ks.append(k)
        calc_k = p_gen.find_scaling((cutout, cutout), screens)

        k_data.append(np.mean(ks))
        calc_data.append(np.mean(calc_k))

    from scipy.stats import ttest_rel

    # Perform dependent two-sample t-test
    t_stat, p_value = ttest_rel(k_data, calc_data)

    print("t-statistic:", t_stat)
    print("p-value:", p_value)

    # print(f"Total Screens: {n_modes}\n\tPhase Size: {phase}\n\tCutout Size: {cutout}\n\tPhase Var: {np.mean(p_vars)}\n\tCutout Var: {np.mean(c_vars)}\n\tScaled Var: {np.mean(scaled_vars)}\n\tScaling Factor: {np.mean(ks)} [{calc_k}]")

    return # END test_phase_variance

def norm(a):
    """
    Normalise an array.
    This uses a standard norm to a range of 0-1
    """
    a = np.array(a)
    mx = a.max()
    mn = a.min()

    # Prevent division by zero
    if mx == mn:
        return a - mn
        
    return (a - mn) / (mx - mn)
    # END norm

def znorm(a):
    """
    Normalise an array.
    This uses Z-score normalisation
    {a - mean(a)}/stdev(a)
    """
    a = np.array(a)
    stdev = a.std()

    # If the stdev is zero, all the data is the same, so a - mean(a) = 0 for all values of a
    if stdev == 0:
        return np.zeros_like(a, dtype=np.float32)
    
    return (a - a.mean()) / stdev
    # END znorm

def regression(X_train, X_test, y_train, y_test, model, name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).flatten()

    df = pd.DataFrame()
    df['prediction'] = y_pred
    df['test'] = y_test.to_numpy()

    print(name)
    print("R² score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("\n")

    sns.lmplot(data=df, x='test', y='prediction')
    plt.plot([df['test'].min(), df['test'].max()], [df['test'].min(), df['test'].max()], 'r:')
    plt.title(name)

    return # END regression

def classification(X_train, X_test, y_train, y_test, model, name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    dfc = pd.DataFrame()
    dfc['prediction'] = y_pred
    dfc['test'] = y_test.to_numpy()

    sns.lmplot(data=dfc, x='test', y='prediction')
    plt.plot([dfc['test'].min(), dfc['test'].max()], [dfc['test'].min(), dfc['test'].max()], 'r:')
    plt.title(name)

    accuracy = accuracy_score(dfc['test'], dfc['prediction'])
    precision = precision_score(dfc['test'], dfc['prediction'])
    recall = recall_score(dfc['test'], dfc['prediction'])

    print(name)
    print(f"Accuracy:  {accuracy:.2f}      Proportion of correct predictions (both positives and negatives.)")
    print(f"Precision: {precision:.2f}      TP / (TP + FP) — How many predicted positives are truly positive?")
    print(f"Recall:    {recall:.2f}      TP / (TP + FN) — How many actual positives were correctly predicted?")

    cm = confusion_matrix(dfc['test'], dfc['prediction'])
    print("Confusion Matrix ([[TN, FP], [FN, TP]]):\n", cm)

    # F1 Score
    f1 = f1_score(dfc['test'], dfc['prediction'])
    print(f"F1 Score: {f1:.2f}      (2 x P x R) / (P + R)\n\n")

    return # END classification

# Machine Learning Regression
class ML_Model():
    def __init__(self, nn_model, loss, cls=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = loss()
        self.model = nn_model()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        self.classifier = cls

        return # END ML_Model.__init__

    def fit(self, X, y):
        X_nn = torch.tensor(X.to_numpy(), dtype=torch.float32)
        y_nn = torch.tensor(y.to_numpy(), dtype=torch.float32)

        if self.classifier:
            self.unique_vals, indices = y_nn.unique(sorted=True, return_inverse=True)
            y_nn = nn.functional.one_hot(indices, num_classes=len(self.unique_vals)).float()

        dataset = TensorDataset(X_nn, y_nn)
        batch_size = 100
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.to(self.device)

        epochs = 16
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = []
            for batch_X, batch_y in loader:
                outputs = self.model(batch_X.to(self.device))
                loss = self.criterion(outputs, batch_y.to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
            # if epoch == 0 or (epoch % 10 == 9):
            print(f"Epoch {epoch+1}/{epochs} complete [{np.mean(epoch_loss)}]")

        return # END ML_Model.fit

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_nn = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
            y_pred = self.model(X_nn.to(self.device)).detach()
        if self.classifier:
            y_pred = self.unique_vals[torch.argmax(y_pred.to('cpu'), dim=1)]

        return y_pred.squeeze().to('cpu').numpy()
        # END ML_Model.predict
    # END_CLASS ML_Model

class ML_AE(ML_Model):
    def encode(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_nn = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
            Z_encode = self.model.encoder(X_nn.to(self.device)).detach()

        return Z_encode.squeeze().to('cpu').numpy()
        # END ML_Model.encode
    
    
