import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import time
import traceback
import joblib
import math
import ast
import os

import pandas as pd
import seaborn as sns
import pyarrow
import uuid
import glob
import pickle

from itertools import combinations
from scipy.stats import gmean, linregress

# Import sklearn models and utility functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
MODEL_FOLDER = 'C:/Code/PhD/Data/Models'

# global queues for different processes
asterism_result_queue = None
asterism_data_queue = None

def run_tests():
    mp.set_start_method('spawn')
    
    cfg = AsterismConfig()
    cfg.N = 35 # 8 radial degrees
    cfg.r0 = 0.15
    n_tests = 500
    n_training = 200

    # for t in ['a', 'b']:
    #     cfg.n_asterisms = n_tests
    #     cfg.n_atm = n_tests
    #     run_asterism_comparison(cfg)
    #     bulk_compile_asterism_data(f'regime_testing_data_35r015_rng_{t}')

    #     cfg.n_asterisms = n_training
    #     cfg.n_atm = n_training
    #     run_asterism_comparison(cfg)
    #     bulk_compile_asterism_data(f'regime_validation_data_35r015_rng_{t}')

    # transform_data(cfg)
    # convert_parquets_to_csv()

    files = glob.glob(os.path.join(DATA_FOLDER, "*_transformed.parquet"))
    files = [f for f in files if 'rng' in f]
    validation_files = [f for f in files if 'validation' in f]
    test_files = [f for f in files if 'testing' in f]
    train_models(test_files, validation_files)

    plt.show()

    return # END run_tests

def convert_parquets_to_csv():
    files = glob.glob(os.path.join(DATA_FOLDER, '*.parquet'))
    if len(files) > 0:
        for filepath in files:
            df0 = pd.read_parquet(filepath)
            file = filepath.split('.')
            csvpath = os.path.join(DATA_FOLDER, file[0] + '.csv')
            df0.to_csv(csvpath, index=False)
            print(f"Converted parquet file {filepath} to csv file {csvpath}")

class AsterismConfig():
    """
    Class of filtering programs
    Defines the main test functions
    """
    def __init__(self):
        # Size of test set
        self.n_asterisms = 512
        self.n_atm = 512

        self.rng = np.random.default_rng()

        # Atmospheric and tomography simulation paramters
        self.N = 21
        self.Z = zernike.Zernike(self.N)
        self.D = 1.8
        self.r0 = 0.15
        self.alts = [1000, 5000, 10000]
        self.ap_pixels = [100]

        # Different data points to compute while running each asterism
        self.tomo_methods = ['glao', 'ao', 'lsq', 'open_loop', 'fitting']

        return # END AsterismConfig.__init__


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

        tomo_cfg = { 'mask': {}, 'glao': {}, 'ao': {}, 'lsq': {} }
        for atm in atm_gens:
            ap = atm.aperture_size
            tomo_cfg['mask'][ap] =  util.circle(atm.aperture_size)
            tomo_cfg['glao'][ap] =  GLAO(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size)
            tomo_cfg['ao'][ap] =    AO(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size)
            tomo_cfg['lsq'][ap] =   LeastSquares(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size, r0=cfg.r0, method='Tinv')
            
        while not asterism_data_queue.empty(): # Asterism Queue
            group_id, asterism_id, asterism = asterism_data_queue.get()
            t0 = time.time()
            
            print(f"[{worker_id:5} | Worker] Starting simulating asterism ({group_id+1}.{asterism_id+1}) / ({cfg.n_asterisms}.10). Time since start: {t0-tstart:.2f}s")

            asterism_specs = [{
                    'rotation': 0,
                    "asterism": asterism,
                }]

            rotation = cfg.rng.uniform(0, 2*np.pi)
            a = asterisms.rotate_asterism(asterism, rotation)
            asterism_specs.append({
                'rotation': rotation,
                "asterism": a,
            })
            
            for ast_spec in asterism_specs:
                for atm in atm_gens:
                    ap = atm.aperture_size
                    mask = tomo_cfg['mask'][ap]
                    target = beacon.Beacon(0, 0, 10, aperture_mask=mask)
                    
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
                                'rotation': ast_spec['rotation'],
                                'aperture_pixels': ap,
                                'tomo_method': tomo,
                            }

                            for sid, star in enumerate(ast):
                                t, g, m = star.to_tuple()
                                data_point[f'star_{sid}_gamma'] = g
                                data_point[f'star_{sid}_theta'] = t
                                data_point[f'star_{sid}_mag'] = m

                            if 'fitting' in tomo:
                                coeffs = target_coeffs
                            elif 'open_loop' in tomo:
                                coeffs = np.zeros_like(target_coeffs)
                            else:
                                coeffs = tomo_cfg[tomo][ap].run(ast, target, ast_coeffs, ptt=target_coeffs[:3])
                            
                            recon = cfg.Z.compose(target_screen.shape, coeffs)
                            sr = strehl.ratio(recon-target_screen, mask=mask)

                            data_point['strehl_ratio'] = sr

                            asterism_result_queue.put(data_point)

            t1 = time.time()
            dt = t1 - t0
            dt_total = t1 - tstart 
            print(f"[{worker_id:5} | Worker] Finished asterism ({group_id+1}.{asterism_id+1}) / ({cfg.n_asterisms}.10). Took {dt:5.2f}s. Total time since start: {dt_total:.2f}s")
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
    manager = mp.Manager()
    asterism_queue = manager.Queue()
    result_queue = manager.Queue()
    rng = np.random.default_rng()
    # all_asterisms = []

    print(f"Generating {cfg.n_asterisms} asterisms")
    groups = asterisms.get_base_asterisms(cfg.n_asterisms)
    for i, asterism_group in tqdm(enumerate(groups)):
        subgroups = combinations(asterism_group, r=3)
        for j, subgroup in enumerate(subgroups):
            raw = [s.copy() for s in subgroup]
            rng.shuffle(raw)
            asterism_queue.put((i, j, raw))
            
    print(f"Generated {cfg.n_asterisms} asterisms")

    print(f"Generating {cfg.n_atm} simulated atmospheres")
    limit = asterisms.LIMITING_ANGLE
    atm_gens = []
    for ap_pix in cfg.ap_pixels:
        atm_gens.append(phase_screen.Atmosphere(cfg.n_atm, cfg.alts, r0=cfg.r0, D=cfg.D, aperture_size=ap_pix, limiting_angle=limit, rng=rng))

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

def bulk_compile_asterism_data(dest_filename):
    files = glob.glob(os.path.join(DATA_FOLDER, 'batch', '*.parquet'))
    if len(files) > 0:
        
        df = pd.DataFrame()

        for f in files:
            df0 = pd.read_parquet(f)
            df = pd.concat([df, df0], ignore_index=True)

        file_path = os.path.join(DATA_FOLDER, dest_filename + f'.parquet')
        df.to_parquet(file_path, index=False)
        
        print(f"Merged data in {os.path.join(DATA_FOLDER, 'batch')} into {file_path}")

        for f in files:
            os.remove(f)
        print(f"Deleted {len(files)} parquet files in {os.path.join(DATA_FOLDER, '/batch')}")
    else:
        print(f"No parquet files in {os.path.join(DATA_FOLDER, 'batch')} to compile into dataframe together")

    return # END run_asterism_comparison

def transform_data(cfg):
    """
    Filter asterisms for good and bad
    """
    def compute_asterism_features(row):
        a = [beacon.Beacon(row[f'star_{i}_theta'], row[f'star_{i}_gamma'], row[f'star_{i}_mag']) for i in range(3)]
        
        data = {
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
            'f_distance_to_closest_edge_obsolete' : asterisms.distance_to_closest_line(a),
            'f_perimeter' : asterisms.perimeter(a),
            'f_min_interior_angle' : asterisms.min_angle(a),
            'f_maximum_interior_angle' : asterisms.max_angle(a),
            'f_gaussian_area_within_triangle' : asterisms.gaussian_area_within_triangle(a),
            'f_overlap' : asterisms.overlap(a, cfg.alts),
            'f_signed_distance_function' : asterisms.is_target_in_triangle(a),
            'f_distance_to_closest_line' : np.abs(asterisms.is_target_in_triangle(a)),
        }
        for i in range(3):
            data[f'star_{i}_theta'] = row[f'star_{i}_theta']
            data[f'star_{i}_gamma'] = row[f'star_{i}_gamma']
            data[f'star_{i}_mag'] = row[f'star_{i}_mag']

        return data

    files = glob.glob(os.path.join(DATA_FOLDER, '*.parquet'))
    files = [f for f in files if "transformed" not in f]
    if len(files) == 0:
        return
    
    files = [f for f in files if '_transformed' not in f]
    # files = [f for f in files if '_rng' in f]
    
    for filepath in files:
        df = pd.read_parquet(filepath)
        file = filepath.split('.')
        savepath = os.path.join(DATA_FOLDER, file[0] + '_transformed.parquet')

        star_cols = []
        for i in range(3):
            star_cols.append(f'star_{i}_gamma')
            star_cols.append(f'star_{i}_theta')
            star_cols.append(f'star_{i}_mag')
        
        df = df[df['rotation'] == 0.0]
        df = df.drop(columns=['aperture_pixels', 'rotation'])

        # Calculate performance relative to baseline (open loop) performance
        baseline_factors = ['group_id', 'asterism_id', 'atmosphere_id']
        baseline = df[df['tomo_method'] == 'open_loop'].copy()
        baseline = baseline.rename(columns={'strehl_ratio': f"strehl_baseline"})
        baseline = baseline.drop(columns=['tomo_method'] + star_cols)  # to allow joining on all ID columns
        df = df.merge(baseline, on=baseline_factors)

        # Copy GLAO and AO performance, for calculating tomographic regime
        baseline_factors = ['group_id', 'asterism_id', 'atmosphere_id']
        baseline = df[df['tomo_method'] == 'glao'].copy()
        baseline = baseline.rename(columns={'strehl_ratio': f"strehl_glao"})
        baseline = baseline.drop(columns=['tomo_method', 'strehl_baseline'] + star_cols)  # to allow joining on all ID columns
        df = df.merge(baseline, on=baseline_factors)

        baseline_factors = ['group_id', 'asterism_id', 'atmosphere_id']
        baseline = df[df['tomo_method'] == 'ao'].copy()
        baseline = baseline.rename(columns={'strehl_ratio': f"strehl_ao"})
        baseline = baseline.drop(columns=['strehl_baseline', 'strehl_glao', 'tomo_method'] + star_cols)  # to allow joining on all ID columns
        df = df[df['tomo_method'] == 'lsq']
        df = df.merge(baseline, on=baseline_factors)

        df = df[df['tomo_method'] == 'lsq']
        for affix in ['ratio']: #, 'glao', 'ao']:
            df[f'strehl_{affix}_rel'] = df[f'strehl_{affix}'] / df[f'strehl_baseline']
            df[f'strehl_{affix}_rel_log'] = (df[f'strehl_{affix}'] / df[f'strehl_baseline']).apply(np.log)



        # Statistical analysis

        # Aggregate results along the atmosphere column.
        result_columns = ['strehl_ratio', 'strehl_glao', 'strehl_ao']
        result_col_rel = [f'{col}_rel{t}' for col in ['strehl_ratio'] for t in ['', '_log']]

        id_factors = ['group_id', 'asterism_id', 'tomo_method'] + star_cols

        agg_dict = {
            col: ['mean'] for col in result_col_rel + result_columns
        }

        df = df.groupby(id_factors)[result_col_rel + result_columns].agg(agg_dict).reset_index()
        df.columns = ['_'.join(col).rstrip('_') if isinstance(col, tuple) else col for col in df.columns]

        baseline_factors = ['group_id', 'asterism_id']
        df['regime'] = 'ao'
        df.loc[df['strehl_glao_mean'] > df['strehl_ao_mean'], 'regime'] = 'glao'
        df.loc[(df['strehl_ratio_mean'] > df['strehl_glao_mean']) & (df['strehl_ratio_mean'] > df['strehl_ao_mean']), 'regime'] = 'tomo'
        
        # df_sorted = df.sort_values(by=["group_id", "asterism_id"])
        # print(df_sorted.head(10))
        # quit()

        ast_df = df[['group_id', 'asterism_id'] + star_cols].copy()
        ast_df = ast_df.drop_duplicates(subset=star_cols)

        # Compute features from asterism data
        df0 = ast_df.apply(compute_asterism_features, axis=1)
        df0 = pd.DataFrame(df0.tolist(), index=df0.index)
        df = df.merge(df0, on=star_cols, how='inner')
        print(df.columns)

        df.to_parquet(savepath, index=False)

def train_models(files, validation_files, data_dir=DATA_FOLDER):

    test_df = pd.DataFrame()
    valid_df = pd.DataFrame()

    for n, file in enumerate(files):
        tdf = pd.read_parquet(os.path.join(data_dir, file))
        tdf['file_id'] = f'test_{n}'

        test_df = pd.concat((test_df, tdf))

    for n, v_file in enumerate(validation_files):
        vdf = pd.read_parquet(os.path.join(data_dir, v_file))
        vdf['file_id'] = f'validate_{n}'
        valid_df = pd.concat((valid_df, vdf))

    features = [
            'f_distance_0',
            'f_distance_1',
            'f_distance_2',
            'f_angle_0',
            'f_angle_1',
            'f_angle_2',
            'f_perimeter',
            'f_area',
            'f_mean_distance',
            'f_signed_distance_function',
            'f_distance_to_closest_line',
            'f_centroid_distance',
            'f_incentre_distance',
            'f_incircle_radius',
            ]
    
    models = {
        "NN": MLPClassifier(hidden_layer_sizes=[32, 32, 32], learning_rate_init=0.01, activation='relu'),
        "boost": AdaBoostClassifier(),
        "random_forest": RandomForestClassifier(),
        # "gaussian": GaussianProcessClassifier(),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(),
        "svm_rbf": SVC(kernel='rbf'),
        "tree": DecisionTreeClassifier()
    }
    # StandardScaler

    id_columns = ['file_id', 'group_id', 'asterism_id']

    print(test_df.columns)
    print(test_df['regime'].unique())
    test_df['tomo_regime'] = test_df['regime'] == 'tomo'
    valid_df['tomo_regime'] = valid_df['regime'] == 'tomo'
    print(test_df['tomo_regime'].value_counts())
    print(valid_df['tomo_regime'].value_counts())

    X_train = test_df[features]
    X_test = valid_df[features]

    y_train = test_df['tomo_regime']
    y_test = valid_df['tomo_regime']

    data = (X_train, X_test, y_train, y_test)

    for name, model in tqdm(models.items()):
        results, trained_model, scaler = run_classification(data, model, name, scale=True)
        filename = os.path.join(MODEL_FOLDER, f'{name}_tomo_regime.pkl')
        print(results)
        with open(filename, 'wb') as f:
            pickle.dump((trained_model, scaler), f)


    # data = []

    # X = df[feature_columns]
    # y =df[y_col]

    # xy_data = train_test_split(X, y, test_size=split_size)

    # model = 
    # data.append()

def run_classification(data, model, name, scale=False):

    X_train, X_test, y_train, y_test = data

    # print(np.isnan(X_train).any())
    # print(np.isnan(y_train).any())

    if scale == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    # print(np.isnan(X_train).any())
    # print(np.isnan(y_train).any())

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    dfc = pd.DataFrame()
    dfc['prediction'] = y_pred
    dfc['test'] = y_test.to_numpy()

    cls_data = {
        'model': name,
        'accuracy': accuracy_score(dfc['test'], dfc['prediction']),
        'precision': precision_score(dfc['test'], dfc['prediction']),
        'recall': recall_score(dfc['test'], dfc['prediction']),
        'f1': f1_score(dfc['test'], dfc['prediction'])
    }

    # sns.lmplot(data=dfc, x='test', y='prediction')
    # plt.plot([dfc['test'].min(), dfc['test'].max()], [dfc['test'].min(), dfc['test'].max()], 'r:')
    # plt.title(name)

    # accuracy = accuracy_score(dfc['test'], dfc['prediction'])
    # precision = precision_score(dfc['test'], dfc['prediction'])
    # recall = recall_score(dfc['test'], dfc['prediction'])

    # print(name)
    # print(f"Accuracy:  {accuracy:.2f}      Proportion of correct predictions (both positives and negatives.)")
    # print(f"Precision: {precision:.2f}      TP / (TP + FP) — How many predicted positives are truly positive?")
    # print(f"Recall:    {recall:.2f}      TP / (TP + FN) — How many actual positives were correctly predicted?")

    # cm = confusion_matrix(dfc['test'], dfc['prediction'])
    # print("Confusion Matrix ([[TN, FP], [FN, TP]]):\n", cm)

    # # F1 Score
    # f1 = f1_score(dfc['test'], dfc['prediction'])
    # print(f"F1 Score: {f1:.2f}      (2 x P x R) / (P + R)\n\n")

    return cls_data, model, scaler