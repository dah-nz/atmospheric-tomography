import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import time
import traceback
import joblib
import math
import ast

import pandas as pd
import seaborn as sns
import pyarrow
import uuid
import glob

from itertools import combinations
from scipy.stats import gmean, linregress

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
from tomography import asterisms, beacon, phase_screen, strehl
from tomography.tomography import GLAO, LeastSquares, get_coeffs_for_asterism, AO
from basis import zernike
import util

SENTINEL = 'STOP'
DATA_FOLDER = 'C:/Code/PhD/Data'

# global queues for different processes
asterism_result_queue = None
asterism_data_queue = None

def run_tests():
    # test_asterism_generation()
    # test_phase_variance()

    # test_asterism_cross_entropy()
    
    cfg = AsterismConfig()
    # run_asterism_comparison(cfg)
    # bulk_compile_asterism_data()
    filter_asterisms(cfg)
    # load_angle_data_test()

    plt.show()

    return # END run_tests

def test_asterism_cross_entropy():
    asts = asterisms.get_base_asterisms(1)

    circ_x = [0, -120, 0, 120]
    circ_y = [120, 0, -120, 0]

    for grp in asts:
        ast_group = asterisms.get_asterisms_from_group(grp)
        baseline = ast_group['star_0_circular']
        
        for i, (n, ast) in enumerate(ast_group.items()):
            idx = i % 9
            if idx == 0:
                fig, axs = plt.subplots(3, 3)
            ce = asterisms.circular_similarity(ast, baseline)
            print(n, ce)
            
            r = idx // 3
            c = idx % 3
            axs[r, c].plot(circ_x, circ_y, alpha=0.0)

            ax, ay = asterisms.get_asterism_coords(ast)
            axs[r, c].plot(ax, ay, 'b-', marker='o')
            bx, by = asterisms.get_asterism_coords(baseline)
            axs[r, c].plot(bx, by, 'r-', marker='+')
            axs[r, c].set_title(f'{n} : {ce:.4f}')
            axs[r, c].set_aspect('equal', 'box')


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

class AsterismConfig():
    """
    Class of filtering programs
    Defines the main test functions
    """
    def __init__(self):
        # Size of test set
        self.n_asterisms = 32
        self.n_atm = 256

        # Atmospheric and tomography simulation paramters
        self.N = 21
        self.Z = zernike.Zernike(self.N)
        self.D = 1.8
        self.r0 = 0.25
        self.alts = [1000, 5000, 10000]
        self.ap_pixels = [100]

        # Different data points to compute while running each asterism
        self.tomo_methods = ['glao', 'lsq_T', 'lsq_K', 'open_loop', 'fitting'] # 'tiptilt', 'ao', 'lsq'

        # Perturbations for all the guide stars
        if True:
            max_perturb = 1.0 # arcseconds
            n_perturbs = 5 # number of perturbations in each direction
            x = np.linspace(-max_perturb, max_perturb, n_perturbs)
            y = np.linspace(-max_perturb, max_perturb, n_perturbs)
            perturbations = np.stack(np.meshgrid(x, y)).T.reshape(-1, 2)
            self.max_full_perturbation = np.amax(np.sqrt(perturbations[:, 0] ** 2 + perturbations[:, 1] ** 2))
            self.full_perturbations = [tuple(p) for p in perturbations]
            self.perturbed_asterism_types = ["_raw", "star_0_circular"]
        else:
            self.max_full_perturbation = 0.0
            self.full_perturbations = [(0.0, 0.0)]
            self.perturbated_asterism_types = []

        # Perturbations for a single guide star
        if True:
            max_perturb = 1 # arcseconds
            n_perturbs = 3 # number of perturbations in each direction
            x = np.linspace(-max_perturb, max_perturb, n_perturbs)
            y = np.linspace(-max_perturb, max_perturb, n_perturbs)
            perturbations = np.stack(np.meshgrid(x, y)).T.reshape(-1, 2)
            self.max_single_perturbation = np.amax(np.sqrt(perturbations[:, 0] ** 2 + perturbations[:, 1] ** 2))
            self.single_perturbations = [tuple(p) for p in perturbations]
            self.single_perturbed_asterism_types = ["_raw", "star_0_circular"]
        else:
            self.max_single_perturbation = 0.0
            self.single_perturbations = [(0.0, 0.0)]

        return # END Filter.__init__


# Global queue reference for workers
def asterism_test_init(result_queue, asterism_queue):
    global asterism_result_queue
    global asterism_data_queue

    asterism_result_queue = result_queue
    asterism_data_queue = asterism_queue

def asterism_test_worker(cfg, atm_gens):
    try:
        tstart = time.time()
        worker_id = os.getpid()
        print(f"[{worker_id:5} | Worker] Started")

        tomo_cfg = { 'mask': {}, 'glao': {}, 'ao': {}, 'lsq': {}, 'lsq_T': {}, 'lsq_K': {} }
        for atm in atm_gens:
            ap = atm.aperture_size
            tomo_cfg['mask'][ap] =  util.circle(atm.aperture_size)
            tomo_cfg['glao'][ap] =  GLAO(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size)
            tomo_cfg['ao'][ap] =    AO(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size)
            tomo_cfg['lsq'][ap] =   LeastSquares(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size, r0=cfg.r0, start_angle=15, end_angle=45)
            tomo_cfg['lsq_T'][ap] = LeastSquares(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size, r0=cfg.r0, method='Tinv')
            tomo_cfg['lsq_K'][ap] = LeastSquares(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size, r0=cfg.r0, method='K')

        target = beacon.Beacon(0, 0, 10, aperture_mask=mask)
            
        while not asterism_data_queue.empty(): # Asterism Queue
            group_id, asterism_id, asterism_type, asterism = asterism_data_queue.get()
            t0 = time.time()
            
            print(f"[{worker_id:5} | Worker] Starting simulating asterism '{asterism_type}' ({group_id+1}.{asterism_id+1}) / ({cfg.n_asterisms}.45). Time since start: {t0-tstart:.2f}s")

            asterism_specs = [
                {
                    "full_perturbation": (0.0, 0.0),
                    "single_perturbation": (0.0, 0.0),
                    "star_perturbed": 0,
                    "asterism": asterism
                }
            ]
            for check_asterism_type in cfg.perturbed_asterism_types:
                if check_asterism_type in asterism_type:
                    for p in cfg.full_perturbations:
                        if p != (0.0, 0.0):
                            asterism_specs.append(
                                {
                                    "full_perturbation": p,
                                    "single_perturbation": (0.0, 0.0),
                                    "star_perturbed": 0,
                                    "asterism": asterisms.perturb_asterism(asterism, p)
                                }
                            )
            for check_asterism_type in cfg.single_perturbed_asterism_types:
                if check_asterism_type in asterism_type:
                    for p in cfg.single_perturbations:
                        if p != (0.0, 0.0):
                            for star, s_ast in asterisms.perturb_asterism_single_star(asterism, p):
                                asterism_specs.append(
                                    {
                                        "full_perturbation": (0.0, 0.0),
                                        "single_perturbation": p,
                                        "star_perturbed": star,
                                        "asterism": s_ast
                                    }
                                )

            for ast_spec in asterism_specs:
                for atm in atm_gens:
                    ap = atm.aperture_size
                    mask = tomo_cfg['mask'][ap]
                    

                    for atm_id, screens in enumerate(atm):
                        ast = ast_spec['asterism']

                        target_coeffs = target.cutout_coeffs(cfg.Z, cfg.D, cfg.alts, screens, atm.metapupil_diameters)
                        target_screen = target.total_cutout(cfg.D, cfg.alts, screens, atm.metapupil_diameters)
                        ast_coeffs = get_coeffs_for_asterism(ast, cfg.Z, cfg.D, screens, cfg.alts, atm.metapupil_diameters)

                        for tomo in cfg.tomo_methods:

                            data_point = {
                                'group_id': group_id,
                                'asterism_id': asterism_id,
                                'atmosphere_id': atm_id,
                                'full_perturbation': ast_spec['full_perturbation'],
                                'single_perturbation': ast_spec['single_perturbation'],
                                'perturbed_star': ast_spec['star_perturbed'],
                                'aperture_pixels': ap,
                                'tomo_method': tomo
                            }

                            if 'lsq' in tomo: # any 'lsq'
                                coeffs = tomo_cfg[tomo][ap].run(ast, target, ast_coeffs, ptt=target_coeffs[:3])
                            elif 'ao' in tomo: # 'glao' and 'ao'
                                coeffs = tomo_cfg[tomo][ap].run(ast, target, ast_coeffs)
                            elif 'tiptilt' in tomo:
                                coeffs = np.zeros_like(target_coeffs)
                                coeffs[:3] = target_coeffs[:3]
                            elif 'fitting' in tomo:
                                coeffs = target_coeffs
                            else: # 'ol'
                                coeffs = np.zeros_like(target_coeffs)

                            recon = cfg.Z.compose(target_screen.shape, coeffs)
                            recon_rmse = util.rmse(recon, target_screen, mask=mask)
                            coeff_rmse = util.rmse(coeffs, target_coeffs)
                            sr = strehl.ratio(recon-target_screen, mask=mask)

                            data_point['recon_rmse'] = recon_rmse
                            data_point['coeff_rmse'] = coeff_rmse
                            data_point['strehl_ratio'] = sr

                            asterism_result_queue.put(data_point)

            t1 = time.time()
            dt = t1 - t0
            dt_total = t1 - tstart 
            print(f"[{worker_id:5} | Worker] Finished asterism ({group_id+1}.{asterism_id+1}) / ({cfg.n_asterisms}.45). Took {dt:5.2f}s. Total time since start: {dt_total:.2f}s")
    except Exception as e:
        print(type(e))    # the exception type
        print(e.args)     # arguments stored in .args
        print(e)
        print(traceback.format_exc(), flush=True)

    print(f'[{worker_id:5} | Worker] Completed. Exiting')
    return True # END asterism_test_worker

def asterism_test_writer(data_q):
    t0 = time.time()
    data_batch = []
    batch_size = 1000000
    output_directory = os.path.join(DATA_FOLDER, 'batch')
    worker_id = os.getpid()
    print(f'[{worker_id:5} | Writer] Started')

    while True:
        item = data_q.get()
        if item == SENTINEL:
            break
        data_batch.append(item)
        
        if len(data_batch) >= batch_size:
            df = pd.DataFrame(data_batch)
            file_path = os.path.join(output_directory, f'batch_{uuid.uuid4()}.parquet')
            df.to_parquet(file_path, index=False, engine='pyarrow')
            print(f'[{worker_id:5} | Writer] Wrote {len(data_batch)} rows to {file_path} [Time since start: {time.time() - t0:.2f}s]')
            data_batch.clear()
    
    if data_batch:
        df = pd.DataFrame(data_batch)
        file_path = os.path.join(output_directory, f'batch_{uuid.uuid4()}.parquet')
        df.to_parquet(file_path, index=False, engine='pyarrow')
        print(f'[{worker_id:5} | Writer] Wrote final {len(data_batch)} rows to {file_path}')
    total_time = time.time() - t0
    print(f'[{worker_id:5} | Writer] Total Time Running: {total_time:.2f}s ({total_time/3600:.2f} hrs)')
    print(f'[{worker_id:5} | Writer] Completed. Exiting')
    return # END Filter.saver

def run_asterism_comparison(cfg):
    """
    Filter.run_asterism_comparison()

    Run a set of asterisms on a set of static phase screens.
    """
    mp.set_start_method('spawn')
    manager = mp.Manager()
    asterism_queue = manager.Queue()
    result_queue = manager.Queue()
    all_asterisms = []

    print(f"Generating {cfg.n_asterisms} asterisms")
    groups = asterisms.get_base_asterisms(cfg.n_asterisms)
    for i, asterism_group in tqdm(enumerate(groups)):
        asts = asterisms.get_asterisms_from_group(asterism_group)
        for j, (ast_type, ast) in enumerate(asts.items()):
            asterism_queue.put((i, j, ast_type, ast))
            ast_data = {
                'group_id': i,
                'asterism_id': j,
                'asterism_type': ast_type
            }
            for k, star in enumerate(ast):
                ast_data[f'star_{k}'] = star.to_tuple()
            all_asterisms.append(ast_data)

    ast_df = pd.DataFrame(all_asterisms)
    ast_df.to_parquet(os.path.join(DATA_FOLDER, 'asterism_geometry_data.parquet'))

    print(f"Generated {cfg.n_asterisms} asterisms")

    print(f"Generating {cfg.n_atm} simulated atmospheres")
    limit = asterisms.LIMITING_ANGLE + np.ceil(cfg.max_full_perturbation+cfg.max_single_perturbation)     # Increase limiting angle for atmosphere generation to account for asterism perturbations.
    atm_gens = []
    for ap_pix in cfg.ap_pixels:
        rng_gen = np.random.default_rng(seed=31415)
        atm_gens.append(phase_screen.Atmosphere(cfg.n_atm, cfg.alts, r0=cfg.r0, D=cfg.D, aperture_size=ap_pix, limiting_angle=limit, rng=rng_gen))

    print(f"Completed simulating {cfg.n_atm} atmospheres")

    n_processes = 8
    print("Run multi-processing for asterism quality filtering")
    writer_proc = mp.Process(target=asterism_test_writer, args=(result_queue,))
    writer_proc.start()

    with manager.Pool(
        processes=n_processes,
        initializer=asterism_test_init,
        initargs=(result_queue, asterism_queue)
    ) as pool:
        for _ in range(n_processes):
            pool.apply_async(asterism_test_worker, args=(cfg, atm_gens))
        
        pool.close()
        pool.join()
        
    result_queue.put(SENTINEL)
    writer_proc.join()

    print("Multi-processing completed")

def bulk_compile_asterism_data():
    files = glob.glob(os.path.join(DATA_FOLDER, 'batch', '*.parquet'))
    if len(files) > 0:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        file_path = os.path.join(DATA_FOLDER, 'asterism_quality_data.parquet')
        df.to_parquet(file_path, index=False)
        print(f"Merged data in {os.path.join(DATA_FOLDER, 'batch')} into {file_path}")
        for f in files:
            os.remove(f)
        print(f"Deleted {len(files)} parquet files in {os.path.join(DATA_FOLDER, '/batch')}")
    else:
        print(f"No parquet files in {os.path.join(DATA_FOLDER, 'batch')} to compile into dataframe together")

    return # END run_asterism_comparison

def filter_asterisms(cfg):
    """
    Filter asterisms for good and bad
    """

    # Build dataframe from asterism quality data
    df = pd.read_parquet(os.path.join(DATA_FOLDER, 'asterism_quality_data.parquet'))
    ast_df = pd.read_parquet(os.path.join(DATA_FOLDER, 'asterism_geometry_data.parquet'))

    def compute_asterism_features(row):
        a = [beacon.Beacon.from_tuple(row[f'star_{i}']) for i in range(3)]
        b = [beacon.Beacon.from_tuple(row[f'star_{i}_baseline']) for i in range(3)]

        return {
            'group_id': row['group_id'],
            'asterism_id': row['asterism_id'],
            'asterism_type': row['asterism_type'],
            'f_distance_0': a[0]._gamma_arcsec,
            'f_distance_1': a[1]._gamma_arcsec,
            'f_distance_2': a[2]._gamma_arcsec,
            'f_angle_0':    a[0].theta,
            'f_angle_1':    a[1].theta,
            'f_angle_2':    a[2].theta,
            'f_mean_distance': asterisms.ave_distance(a),
            'f_min_distance' : asterisms.min_distance(a),
            'f_max_distance' : asterisms.max_distance(a),
            'f_area' : asterisms.area(a),
            'f_centroid_distance' : asterisms.centroid_distance(a),
            'f_circumcentre_distance' : asterisms.circumcentre_distance(a),
            'f_circumcircle_radius' : asterisms.circumcircle_radius(a),
            'f_length_ratio' : asterisms.length_ratio(a),
            'f_average_side_length' : asterisms.average_side_length(a),
            'f_minimum_side_length' : asterisms.minimum_side_length(a),
            'f_maximum_side_length' : asterisms.maximum_side_length(a),
            'f_incentre_distance' : asterisms.incentre_distance(a),
            'f_incircle_radius' : asterisms.incircle_radius(a),
            'f_perimeter_area_ratio' : asterisms.perimeter_area_ratio(a),
            'f_closest_furthest_side_length' : asterisms.closest_furthest_side_length(a),
            'f_distance_to_closest_edge' : asterisms.distance_to_closest_line(a),
            'f_perimeter' : asterisms.perimeter(a),
            'f_min_interior_angle' : asterisms.min_angle(a),
            'f_maximum_interior_angle' : asterisms.max_angle(a),
            'f_gaussian_area_within_triangle' : asterisms.gaussian_area_within_triangle(a),
            'f_overlap' : asterisms.overlap(a, cfg.alts),
            'f_target_in_triangle' : asterisms.is_target_in_triangle(a),
            'f_circular_similarity': asterisms.circular_similarity(a, b),
        }

    print(df.shape[0])

    perturbation_analysis(cfg, df, ast_df)

    methods_to_compare = ['lsq_T', 'lsq_K']
    df = df[(df['full_perturbation'].apply(lambda x: np.array_equal(x, np.array([0.0, 0.0])))) 
            & (df['single_perturbation'].apply(lambda x: np.array_equal(x, np.array([0.0, 0.0]))))
            & (df['aperture_pixels'] == 100)]

    df = df.drop(columns=['full_perturbation', 'single_perturbation', 'perturbed_star'])
    
    id_factors = ['group_id', 'asterism_id', 'atmosphere_id']
    result_columns = ['recon_rmse', 'coeff_rmse', 'strehl_ratio']

    baseline = df[df['tomo_method'] == 'open_loop'].copy()
    baseline = baseline.rename(columns={col: f"{col}_ol_baseline" for col in result_columns})
    baseline = baseline.drop(columns=['tomo_method', 'aperture_pixels'])  # to allow joining on all ID columns """, 'singular_values_ce', 'singular_values_geomean', 'singular_values_ratio', 'singular_values_min'"""

    df = df.merge(baseline, on=id_factors)

    for col in result_columns:
        df[f'{col}_ol_relative'] = df[col] / df[f'{col}_ol_baseline']

    df = df[df['tomo_method'].isin(methods_to_compare)]

    print(df.shape[0])

    baseline_asts = ast_df[ast_df['asterism_type'] == 'star_0_circular'].copy()
    star_cols = ['star_0', 'star_1', 'star_2']
    baseline_asts = baseline_asts.rename(columns={col: f'{col}_baseline' for col in star_cols})
    baseline_asts = baseline_asts.drop(columns=['asterism_id', 'asterism_type'])
    ast_df = ast_df.merge(baseline_asts, on=['group_id'])

    df0 = ast_df.apply(compute_asterism_features, axis=1)
    df0 = pd.DataFrame(df0.tolist(), index=df0.index)
    df = df.merge(df0, how='inner')
    # df = pd.concat([df, df0], axis=1)

    id_factors = ['group_id', 'atmosphere_id', 'tomo_method']
    result_columns = ['recon_rmse', 'coeff_rmse', 'strehl_ratio']

    baseline = df[df['asterism_type'] == 'star_0_circular'].copy()
    baseline = baseline[id_factors+result_columns]  # to allow joining on all ID columns
    baseline = baseline.rename(columns={col: f"{col}_circular_baseline" for col in result_columns})
    df = df.merge(baseline, on=id_factors)

    for col in result_columns:
        df[f'{col}_circular_relative'] = df[col] / df[f'{col}_circular_baseline']
    
    feature_columns = [c for c in df.columns if c.startswith(("singular_values_", "f_"))]
    n_features = len(feature_columns)
    print(f"Number of features: {n_features}")

    # plot_features(df, feature_columns)
    # comparison(df, feature_columns)



    # dimensionality_reduction(df, feature_columns)
    # self.dimensionality_reduction_ML(df0)

    # data_correlation(df)

    # self.run_classification_models(df0)
    # self.run_ML_classification(df0)

    return # END Filter.filter_asterisms

def perturbation_analysis(cfg, df, ast_df):
    tomo_methods = ['glao', 'lsq_T', 'lsq_K']
    metrics = ['coeff_rmse', 'recon_rmse']

    df_p = df.copy()
    
    # Dropping duplicates is very important here - this ensures a single mapping, not multiple between each id and type
    df_p = df_p.merge(ast_df[['asterism_id', 'asterism_type']].drop_duplicates(), on='asterism_id', how='inner')

    df_s = df_p.copy()

    df_p = df_p[df_p['single_perturbation'].apply(lambda x: np.array_equal(x, np.array([0.0, 0.0])))]
    df_p = df_p.drop(columns=['single_perturbation', 'perturbed_star'])
    perturbed_methods = ['lsq_T', 'lsq_K', 'glao']
    df_p = df_p[df_p['tomo_method'].isin(perturbed_methods)]
    mask = False
    for ast_type in cfg.perturbed_asterism_types:
        mask |= df_p['asterism_type'].str.contains(ast_type)
    df_p = df_p[mask]

    df_s = df_s[df_s['full_perturbation'].apply(lambda x: np.array_equal(x, np.array([0.0, 0.0])))]
    df_s = df_s.drop(columns=['full_perturbation'])
    df_s = df_s[df_s['tomo_method'].isin(perturbed_methods)]
    mask = False
    for ast_type in cfg.single_perturbed_asterism_types:
        mask |= df_s['asterism_type'].str.contains(ast_type)
    df_s = df_s[mask]

    plots = [(method, metric) for method in tomo_methods for metric in metrics]

    # errors = {}
    for metric in metrics:
        baseline = df_p[df_p['full_perturbation'].apply(lambda x: np.array_equal(x, np.array([0.0, 0.0])))]
        baseline = baseline[['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id', metric]]
        baseline = baseline.rename(columns={metric: f'{metric}_baseline'})
        df_p = df_p.merge(baseline, on=['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id'], how='left')
        df_p[f'{metric}_diff'] = df_p[metric] - df_p[f'{metric}_baseline']

        baseline = df_s[df_s['single_perturbation'].apply(lambda x: np.array_equal(x, np.array([0.0, 0.0])))]
        baseline = baseline[['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id', 'perturbed_star', metric]]
        baseline = baseline.rename(columns={metric: f'{metric}_baseline'})
        df_s = df_s.merge(baseline, on=['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id', 'perturbed_star'], how='left')
        df_s[f'{metric}_diff'] = df_s[metric] - df_s[f'{metric}_baseline']

    df_p['perturbation'] = df_p['full_perturbation'].apply(lambda x: tuple(x))
    df_p = df_p.drop(columns=['full_perturbation'])
    df0 = df_p.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation'])[[metric for metric in metrics]].mean().reset_index()
    df_p = df_p.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation'])[[f'{metric}_diff' for metric in metrics]].mean().reset_index()
    df_p = df_p.merge(df0, on=['group_id', 'asterism_id', 'tomo_method', 'perturbation'])

    df_s['perturbation'] = df_s['single_perturbation'].apply(lambda x: tuple(x))
    df_s = df_s.drop(columns=['single_perturbation', 'perturbed_star'])
    df0 = df_s.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation'])[[metric for metric in metrics]].mean().reset_index()
    df_s = df_s.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation'])[[f'{metric}_diff' for metric in metrics]].mean().reset_index()
    df_s = df_s.merge(df0, on=['group_id', 'asterism_id', 'tomo_method', 'perturbation'])

    print(df_p[:5])
    print(df_s[:5])

    for method, metric in plots:  
        fig, axs = plt.subplots(5, 5)
        f2, axs2 = plt.subplots(5, 5)
        fig.suptitle(f'Perturbation of {method}: {metric}')
        f2.suptitle(f'Perturbation of {method}: {metric} (Outliers Removed)')

        sfig, saxs = plt.subplots(3, 3)
        sfig.suptitle(f'Single star perturbation of {method}: {metric}')
        
        # Column labels
        for j, label in enumerate(['-1.0"', '-0.5"', '0.0"', '0.5"', '1.0"']):
            fig.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)
            f2.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)

        # Row labels
        for i, label in enumerate(['1.0"', '0.5"', '0.0"', '-0.5"', '-1.0"']):
            fig.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)
            f2.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)

        for j, label in enumerate(['-0.5"', '0.0"', '0.5"']):
            sfig.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)

        for i, label in enumerate(['0.5"', '0.0"', '-0.5"']):
            sfig.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)

        for perturbation in df_p['perturbation'].unique():
            if perturbation != (0.0, 0.0):
                # pert = eval(perturbation)

                errors = df_p[(df_p['perturbation'] == perturbation) & (df_p['tomo_method'] == method)][f"{metric}_diff"].tolist()
                errors = np.array(errors)
                outlier_mask = util.is_outlier(errors)
                outliers = errors[outlier_mask]
                errors_no_outliers = list(errors[~outlier_mask])
                print(f"{method}, {metric}, {perturbation} has {len(outliers)} outliers/{len(outlier_mask)} values")


                c = int(perturbation[0] * 2. + 2.)
                r = int(2. - perturbation[1] * 2.)
                axs[r, c].hist(errors, bins=30, color='g')
                
                c = int(perturbation[0] * 2. + 2.)
                r = int(2. - perturbation[1] * 2.)
                axs2[r, c].hist(errors_no_outliers, bins=30, color='g')
                
            else:
                values = df_p[(df_p['perturbation'] == perturbation) & (df_p['tomo_method'] == method)][metric].tolist()
                axs[2, 2].hist(values, bins=30, color='r')
                axs2[2, 2].hist(values, bins=30, color='r')
            
        for i in range(5):
            axs[4, i].set_xlabel('RMSE (rads)')
            axs[i, 0].set_ylabel('Count')

            axs2[4, i].set_xlabel('RMSE (rads)')
            axs2[i, 0].set_ylabel('Count')

        for perturbation in df_s['perturbation'].unique():
            if perturbation != (0.0, 0.0):

                errors = df_s[(df_s['perturbation'] == perturbation) & (df_s['tomo_method'] == method)][f"{metric}_diff"].tolist()

                c = int(perturbation[0] + 1.)
                r = int(1. - perturbation[1])
                saxs[r, c].hist(errors, bins=10, color='b')
            else:
                values = df_s[(df_s['perturbation'] == perturbation) & (df_s['tomo_method'] == method)][metric].tolist()
                saxs[1, 1].hist(values, bins=10, color='r')

        for i in range(3):
            saxs[2, i].set_xlabel('RMSE (rads)')
            saxs[i, 0].set_ylabel('Count')
        
            
        # fig.tight_layout()
    
    # fig, axs = plt.subplots(5, 5)
    # fig.suptitle("Perturbation measurement error")

    # # Column labels
    # for j, label in enumerate(['-1.0"', '-0.5"', '0.0"', '0.5"', '1.0"']):
    #     fig.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)

    # # Row labels
    # for i, label in enumerate(['1.0"', '0.5"', '0.0"', '-0.5"', '-1.0"']):
    #     fig.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)

    # def to_arr(s):
    #     array = np.array
    #     float32 = np.float32
    #     return eval(s)

        # import re
        # s=re.sub('\[ +', '[', s.strip())
        # # Replace commas and spaces
        # s=re.sub('[,\s]+', ', ', s)
        # return np.array(ast.literal_eval(s))

    # baseline_coeffs = df[df['perturbation'] == "(0.0, 0.0)"]["asterism_coeffs"].tolist()
    # baseline_coeffs = np.array([to_arr(b) for b in baseline_coeffs])
    # baseline_coeffs = baseline_coeffs.reshape(baseline_coeffs.shape[0], -1)

    # for perturbation in df['perturbation'].unique():
    #     if perturbation != "(0.0, 0.0)":
    #         pert = eval(perturbation)

    #         coeffs = df[df['perturbation'] == perturbation]["asterism_coeffs"].tolist()
    #         coeffs = np.array([to_arr(b) for b in coeffs])
    #         coeffs = coeffs.reshape(coeffs.shape[0], -1)

    #         errors = [util.rmse(a, b) for a, b in zip(coeffs, baseline_coeffs)]

    #         c = int(pert[0] * 2. + 2.)
    #         r = int(2. - pert[1] * 2.)
    #         axs[r, c].hist(errors, bins=30, color='b')
    #     else:
    #         coeffs = df[df['perturbation'] == perturbation]["asterism_coeffs"].tolist()
    #         coeffs = np.array([to_arr(b) for b in coeffs])
    #         print(coeffs.shape)
    #         axs[2, 2].hist(np.sqrt(np.mean(coeffs*coeffs, axis=(1, 2))), bins=30, color='r')

    # for i in range(5):
    #     axs[4, i].set_xlabel('ΔRMSE (rads)')
    #     axs[i, 0].set_ylabel('Count')


def comparison(df, features):
    group_ids = df['group_id'].unique().tolist()
    asterism_ids = df['asterism_id'].unique().tolist()

    performance = 'recon_relative_rmse'

    group_asterism_ids = [(gid, aid) for gid in group_ids for aid in asterism_ids]

    total = math.comb(len(group_asterism_ids), 2)

    pairs = combinations(group_asterism_ids, 2)

    data = {}
    for feature in features:
        data[feature] = { 'greater': 0, 'less than': 0, 'equal': 0, 'equal performance': 0 }

    if total > 1000:
        total = 1000
    eps = 1e-3

    rng = np.random.default_rng()
    pairs = rng.choice(list(pairs), total)

    print(len(group_asterism_ids), total)

    for ga0, ga1 in tqdm(pairs, total=total):
        g0, a0 = ga0
        g1, a1 = ga1

        p0 = df.loc[(df['group_id'] == g0) & (df['asterism_id'] == a0), performance].values[0]
        p1 = df.loc[(df['group_id'] == g1) & (df['asterism_id'] == a1), performance].values[0]
        for feature in features:
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
    
    plt.figure()
    # X locations for the groups
    x = np.arange(len(features))  # [0, 1, 2]
    width = 0.7 / 4  # width of each bar

    gt = [data[f]['greater'] for f in features]
    lt = [data[f]['less than'] for f in features]
    eq = [data[f]['equal'] for f in features]
    ep = [data[f]['equal performance'] for f in features]

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
    plt.xticks(x, features, rotation=90)
    plt.legend()
    plt.tight_layout()


def plot_features(df, features):

    asterism_group = {f'star_{i}_circular': 'Single Star Circular' for i in range(5)}
    for i in range(10):
        asterism_group[f'sg{i}_raw'] = f'Raw'
        asterism_group[f'sg{i}_equal_angle'] = f'Equal Angle'
        asterism_group[f'sg{i}_mean_dist'] = f'Mean Distance'
        asterism_group[f'sg{i}_mean_circular'] = f'Mean Circular'

    df['Asterism Group'] = df['asterism_type'].map(asterism_group)

    # df_stars = df[df['asterism_type'].str.contains('star_')]
    # df_sg = [df[df['asterism_type'].str.contains(f'sg{i}_')] for i in range(10)]

    # df_list = df_sg + [df_stars]
    
    for Y in df.columns[df.columns.str.contains('recon') & df.columns.str.contains('relative')]:
        for feature in features:
            if False:
                for ap in df['aperture_pixels'].unique():
                    for tomo in df['tomo_method'].unique():
                        try:
                            df0 = df[(df['aperture_pixels'] == ap) & (df['tomo_method'] == tomo)]
                            out = linregress(df0[feature], df0[Y])
                            print(f"Tomo: {tomo}, Output Type: {Y}, Feature: {feature}: R^2 = {out.rvalue}")
                        except ValueError:
                            print(f"Tomo: {tomo}, Output Type: {Y}, Feature: {feature}: All values are the same ({df0[feature].mean()})")

            sns.lmplot(df, x=feature, y=Y, hue='Asterism Group', col='tomo_method')

def dimensionality_reduction(df, feature_list):
    X = df[feature_list]
    y = df['recon_rmse_ol_relative']

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

    # n_components = 10
    # if n_components < len(feature_list):
    #     # Display higher dimensional data as a PCA -> t-SNE analysis
    #     print("Calculating PCA->t-SNE 2D analysis")
    #     pca = PCA(n_components=n_components)
    #     X_pca = pca.fit_transform(X)

    #     tsne = TSNE(n_components=2)
    #     X_tsne = tsne.fit_transform(X_pca)

    #     plt.figure()
    #     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    #     plt.title("PCA -> t-SNE 2D")

    #     print("Calculating PCA->UMAP 2D analysis")
    #     pca = PCA(n_components=n_components)
    #     X_pca = pca.fit_transform(X)

    #     tsne = umap.UMAP(n_components=2)
    #     X_tsne = tsne.fit_transform(X_pca)

    #     plt.figure()
    #     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    #     plt.title("PCA -> UMAP 2D")

    return # END Filter.dimensionality_reduction
    
def data_correlation(df):
    # Show correlations between variables
    result_columns = ['recon_rmse', 'coeff_rmse']

    df0 = df.filter(regex='^f_.*')
    feature_columns = list(df0.columns)

    plt.figure()

    correlations = df[result_columns + feature_columns].corr().loc[feature_columns, result_columns]
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


def test_asterism_generation():

    all_asts = []
    t0 = time.time()
    groups = asterisms.get_base_asterisms(2)
    t1 = time.time()
    for group in groups:
        asts = asterisms.get_asterisms_from_group(group)
        all_asts = all_asts + asts
    t1 = time.time()
    
    print(f"Time: {t1-t0}")
    print(f"Groups: {len(groups)}; Asterisms: {len(all_asts)}")
    for ast in all_asts:
        print(ast)

    # fixed_asts =   asterisms.get_test_asterisms(100, fixed_rate=1.0, uniform_rate=0.0, offset_rate=0.0)
    # uniform_asts = asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=1.0, offset_rate=0.0)
    # offset_asts =  asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=0.0, offset_rate=1.0)
    # real_asts =    asterisms.get_test_asterisms(100, fixed_rate=0.0, uniform_rate=0.0, offset_rate=0.0)
    # mixed_asts =   asterisms.get_test_asterisms(100, fixed_rate=0.1, uniform_rate=0.2, offset_rate=0.5)
    # mix_2_asts =   asterisms.get_test_asterisms(100, fixed_rate=0.2, uniform_rate=0.2, offset_rate=0.4)
    

    # def getCircle(radius, n_points, centre=(0,0), angle=2*np.pi):
    #     circle = [util.pol2cart(radius, t/n_points*angle) for t in range(n_points + 1)]
    #     X = [x+centre[0] for x, _ in circle]
    #     Y = [y+centre[1] for _, y in circle]
    #     return X, Y
    
    # xcirc, ycirc = getCircle(120, 64)
    
    # fig, axs = plt.subplots(2, 3)
    # fig.suptitle("Asterism Distribution")

    # def plot_asterisms(ax, title, asts):
    #     ax.set_title(title)
    #     ax.set_aspect('equal')
    #     ax.set_xlabel('Arcseconds')
    #     ax.set_ylabel('Arcseconds')
    #     for ast in asts:
    #         x, y = asterisms.get_asterism_coords(ast, theta_offset=True)
    #         ax.scatter(x, y, c='r', alpha=0.2)
    #     for ast in asts:
    #         x, y = asterisms.get_asterism_coords(ast)
    #         x = np.append(x, x[0])
    #         y = np.append(y, y[0])
    #         ax.plot(x, y, 'b', alpha=0.1)
    #     ax.plot(xcirc, ycirc, 'k:', alpha=0.2)

    # plot_asterisms(axs[0, 0], "Fixed Asterisms", fixed_asts)
    # plot_asterisms(axs[0, 1], "Uniform Asterisms", uniform_asts)
    # plot_asterisms(axs[1, 0], "Offset Asterisms", offset_asts)
    # plot_asterisms(axs[1, 1], "Real Asterisms", real_asts)
    # plot_asterisms(axs[0, 2], "Mix 1 Asterisms", mixed_asts)
    # plot_asterisms(axs[1, 2], "Mix 2 Asterisms", mix_2_asts)

    # fig.tight_layout()

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
    
    
