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
    mp.set_start_method('spawn')

    # test_asterism_features()

    # test_asterism_generation()
    # test_phase_variance()

    # test_asterism_cross_entropy()
    
    cfg = AsterismConfig()
    cfg.N = 21
    cfg.n_asterisms = 512
    cfg.n_atm = 512
    cfg.augmented_asterisms = False
    run_asterism_comparison(cfg)
    bulk_compile_asterism_data('testing_data_21r015b')

    cfg.augmented_asterisms = False
    cfg.n_asterisms = 256
    cfg.n_atm = 256
    run_asterism_comparison(cfg)
    bulk_compile_asterism_data('validation_data_21r015b')

    # transform_data(cfg)
    # display_results()
    # load_angle_data_test()

    plt.show()

    return # END run_tests

def test_asterism_features():
    # Print features for Latex document
    stars = [(1.15, 65), (1.25, 280), (1.7, 195)]
    asterism = [beacon.Beacon(t * np.pi/180, g) for g, t in stars]

    for i, s in enumerate(asterism):
        c = (bytes([ord('a') + i])).decode('ascii')[0]
        g = s._gamma_arcsec
        t = s.theta * 180/np.pi
        print(f"\\newcommand{{\\GS{c}g}}{{{g}}}")
        print(f"\\newcommand{{\\GS{c}t}}{{{t}}}")
        
        x, y = s.position_arcsec()
        print(f"\\newcommand{{\\GS{c}x}}{{{x}}}")
        print(f"\\newcommand{{\\GS{c}y}}{{{y}}}")

    incentre = asterisms.incentre(asterism)
    theta = incentre.theta * 180/np.pi
    gamma = incentre._gamma_arcsec

    print(f"\\newcommand{{\\IncentreTheta}}{{{theta}}}")
    print(f"\\newcommand{{\\IncentreGamma}}{{{gamma}}}")

    x, y = incentre.position_arcsec()
    print(f"\\newcommand{{\\IncentreX}}{{{x}}}")
    print(f"\\newcommand{{\\IncentreY}}{{{y}}}")

    radius = asterisms.incircle_radius(asterism)
    print(f"\\newcommand{{\\IncircleRadius}}{{{radius}}}")

    x, y = util.closest_point_to_line(incentre.position_arcsec(), asterism[0].position_arcsec(), asterism[2].position_arcsec())
    print(f"\\newcommand{{\\IncircleInterceptX}}{{{x}}}")
    print(f"\\newcommand{{\\IncircleInterceptY}}{{{y}}}")

    circumcentre = asterisms.circumcentre(asterism)
    theta = circumcentre.theta * 180/np.pi
    gamma = circumcentre._gamma_arcsec
    print(f"\\newcommand{{\\CircumcentreTheta}}{{{theta}}}")
    print(f"\\newcommand{{\\CircumcentreGamma}}{{{gamma}}}")

    radius = asterisms.circumcircle_radius(asterism)
    print(f"\\newcommand{{\\CircumcircleRadius}}{{{radius}}}")

    b = asterisms.centroid(asterism)
    x, y = b.position_arcsec()
    print(f"\\newcommand{{\\CentroidX}}{{{x}}}")
    print(f"\\newcommand{{\\CentroidY}}{{{y}}}")

    xa, ya = asterism[0].position_arcsec()
    xb, yb = asterism[1].position_arcsec()
    xc, yc = asterism[2].position_arcsec()

    print(f"\\newcommand{{\\MidABX}}{{{(xa + xb) / 2}}}")
    print(f"\\newcommand{{\\MidABY}}{{{(ya + yb) / 2}}}")

    print(f"\\newcommand{{\\MidACX}}{{{(xa + xc) / 2}}}")
    print(f"\\newcommand{{\\MidACY}}{{{(ya + yc) / 2}}}")

    print(f"\\newcommand{{\\MidBCX}}{{{(xb + xc) / 2}}}")
    print(f"\\newcommand{{\\MidBCY}}{{{(yb + yc) / 2}}}")

    xab, yab = util.closest_point_to_line((0, 0), asterism[0].position_arcsec(), asterism[1].position_arcsec())
    xac, yac = util.closest_point_to_line((0, 0), asterism[0].position_arcsec(), asterism[2].position_arcsec())
    xbc, ybc = util.closest_point_to_line((0, 0), asterism[1].position_arcsec(), asterism[2].position_arcsec())
    dab = xab**2+yab**2
    dac = xac**2+yac**2
    dbc = xbc**2+ybc**2
    if dab <= min(dab, dac, dbc):
        ab_text = "Closest"
        ac_text = "NotClosestA"
        bc_text = "NotClosestB"
    elif dac <= min(dab, dac, dbc):
        ab_text = "NotClosestA"
        ac_text = "Closest"
        bc_text = "NotClosestB"
    else:
        ab_text = "NotClosestA"
        ac_text = "NotClosestB"
        bc_text = "Closest"

    print(f"\\newcommand{{\\{ab_text}X}}{{{xab}}}")
    print(f"\\newcommand{{\\{ab_text}Y}}{{{yab}}}")

    print(f"\\newcommand{{\\{ac_text}X}}{{{xac}}}")
    print(f"\\newcommand{{\\{ac_text}Y}}{{{yac}}}")

    print(f"\\newcommand{{\\{bc_text}X}}{{{xbc}}}")
    print(f"\\newcommand{{\\{bc_text}Y}}{{{ybc}}}")

    resolution = 256
    dim = 2
    x = np.linspace(-dim, dim, resolution)
    y = np.linspace(-dim, dim, resolution)

    X, Y = np.meshgrid(x, y)
    A = asterism[0].position_arcsec()
    B = asterism[1].position_arcsec()
    C = asterism[2].position_arcsec()
    triangle = np.array([A, B, C])

    def sdf(X, Y):
        field = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                p = np.array([X[i, j], Y[i, j]])
                field[i, j] = util.signed_distance_triangle(p, A, B, C)
        return field * util.circle(resolution)
    
    def point_in_tri(X, Y):
        field = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                p = np.array([X[i, j], Y[i, j]])
                field[i, j] = util.point_in_triangle(A, B, C, p)
        return field * util.circle(resolution)
    
    df = sdf(X, Y)
    vlim=max(np.abs(np.amax(df)), np.abs(np.amin(df)))

    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#33BDF2', "#ACE8FF", '#FFFFFF', '#FFBF7F', "#FF9429"]
    ccmap = LinearSegmentedColormap.from_list('Custom', colors)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(df, extent=(-dim, dim, -dim, dim), origin='lower', cmap=ccmap, vmax=vlim, vmin=-vlim)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Signed Distance')

    plt.show()



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
        self.n_asterisms = 512
        self.n_atm = 512

        # Atmospheric and tomography simulation paramters
        self.N = 21
        self.Z = zernike.Zernike(self.N)
        self.D = 1.8
        self.r0 = 0.15
        self.alts = [1000, 5000, 10000]
        self.ap_pixels = [100]

        # Different data points to compute while running each asterism
        self.tomo_methods = ['glao', 'lsq_T', 'lsq_K', 'open_loop', 'fitting'] # 'tiptilt', 'ao', 'lsq'

        # Include augmented asterisms at asterism generation
        self.augmented_asterisms = True

        # Add noise to position/WFS measurements
        self.position_n = 0
        self.position_uncertainty = 0.5 # arcseconds
        self.single_uncertainty = 0.2 # arcseconds
        self.measurement_n = 0
        self.measurement_uncertainty = 0.1 # 10% of Zernike covariance

        self.perturbation_analysis = False
        # Perturbations for all the guide stars
        if False:
            max_perturb = 0.1 # arcseconds
            self.n_full_perturbs = 5 # number of perturbations in each direction
            self.full_perturb_x = np.array([-1.0, -0.1, 0.0, 0.1, 1.0]) # np.linspace(-max_perturb, max_perturb, self.n_full_perturbs)
            self.full_perturb_y = np.array([-1.0, -0.1, 0.0, 0.1, 1.0]) # np.linspace(-max_perturb, max_perturb, self.n_full_perturbs)
            perturbations = np.stack(np.meshgrid(self.full_perturb_x, self.full_perturb_y)).T.reshape(-1, 2)
            self.max_full_perturbation = np.amax(np.sqrt(perturbations[:, 0] ** 2 + perturbations[:, 1] ** 2))
            # self.full_perturbations = [tuple(p) for p in perturbations]
            self.perturbed_asterism_types = ["_raw", "star_0_circular"]
        else:
            self.max_full_perturbation = 0.0
            self.full_perturbations = [(0.0, 0.0)]
            self.perturbed_asterism_types = []

        # Perturbations for a single guide star
        if False:
            max_perturb = 0.1 # arcseconds
            self.n_single_perturbs = 3 # number of perturbations in each direction
            self.single_perturb_x = np.array([-0.25, 0, 0.25]) # np.linspace(-max_perturb, max_perturb, self.n_single_perturbs)
            self.single_perturb_y = np.array([-0.25, 0, 0.25]) # np.linspace(-max_perturb, max_perturb, self.n_single_perturbs)
            perturbations = np.stack(np.meshgrid(self.single_perturb_x, self.single_perturb_y)).T.reshape(-1, 2)
            self.max_single_perturbation = np.amax(np.sqrt(perturbations[:, 0] ** 2 + perturbations[:, 1] ** 2))
            # self.single_perturbations = [tuple(p) for p in perturbations]
            self.single_perturbed_asterism_types = ["_raw", "star_0_circular"]
        else:
            self.max_single_perturbation = 0.0
            self.single_perturbations = [(0.0, 0.0)]
            self.single_perturbed_asterism_types = []

        if False:
            self.rotations = np.linspace(0, 2*np.pi/3, 4)
            self.rotated_asterism_types = ["_raw", "star_0_circular"]
        else:
            self.rotations = np.array([])
            self.rotated_asterism_types = []


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

        tomo_cfg = { 'mask': {}, 'glao': {}, 'ao': {}, 'lsq_T': {}, 'lsq_K': {} }
        for atm in atm_gens:
            ap = atm.aperture_size
            tomo_cfg['mask'][ap] =  util.circle(atm.aperture_size)
            tomo_cfg['glao'][ap] =  GLAO(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size)
            tomo_cfg['ao'][ap] =    AO(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size)
            # tomo_cfg['lsq'][ap] =   LeastSquares(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size, r0=cfg.r0, start_angle=15, end_angle=45)
            tomo_cfg['lsq_T'][ap] = LeastSquares(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size, r0=cfg.r0, method='Tinv')
            tomo_cfg['lsq_K'][ap] = LeastSquares(cfg.Z, cfg.D, cfg.N, cfg.alts, atm.aperture_size, r0=cfg.r0, method='K')

        # Covariance matrix for adding measurement noise
        C_mat = tomo_cfg['lsq_T'][ap].C_diag * cfg.measurement_uncertainty

        while not asterism_data_queue.empty(): # Asterism Queue
            group_id, asterism_id, asterism_type, asterism = asterism_data_queue.get()
            t0 = time.time()
            
            print(f"[{worker_id:5} | Worker] Starting simulating asterism '{asterism_type}' ({group_id+1}.{asterism_id+1}) / ({cfg.n_asterisms}.45). Time since start: {t0-tstart:.2f}s")

            asterism_specs = [{
                    "perturbation_type": 'none',
                    "perturbation_x": 0.0,
                    "perturbation_y": 0.0,
                    "star_perturbed": 0,
                    'rotation': 0,
                    "asterism": asterism,
                    'noise': 'none',
                }]

            for check_asterism_type in cfg.perturbed_asterism_types:
                if check_asterism_type in asterism_type:
                    for p_x in cfg.full_perturb_x:
                        for p_y in cfg.full_perturb_y:
                            a = asterisms.perturb_asterism(asterism, (p_x, p_y))
                            asterism_specs.append(
                                {
                                    "perturbation_type": 'full',
                                    "perturbation_x": p_x,
                                    "perturbation_y": p_y,
                                    "star_perturbed": 0,
                                    'rotation': 0,
                                    "asterism": a,
                                    'noise': 'none',
                                }
                            )

            for check_asterism_type in cfg.single_perturbed_asterism_types:
                if check_asterism_type in asterism_type:
                    for p_x in cfg.single_perturb_x:
                        for p_y in cfg.single_perturb_y:
                            for star, s_ast in asterisms.perturb_asterism_single_star(asterism, (p_x, p_y)):
                                asterism_specs.append(
                                    {
                                    "perturbation_type": 'single',
                                    "perturbation_x": p_x,
                                    "perturbation_y": p_y,
                                    "star_perturbed": star,
                                    'rotation': 0,
                                    "asterism": s_ast,
                                    'noise': 'none',
                                    }
                                )

            for check_asterism_type in cfg.rotated_asterism_types:
                if check_asterism_type in asterism_type:
                    for rotation in cfg.rotations:
                        a = asterisms.rotate_asterism(asterism, rotation)
                        asterism_specs.append({
                            'perturbation_type': 'rotate',
                            "perturbation_x": 0,
                            "perturbation_y": 0,
                            "star_perturbed": 0,
                            'rotation': rotation,
                            "asterism": a,
                            'noise': 'none',
                        })

            
            if cfg.position_n > 0:
                for _ in range(cfg.position_n):
                    asterism_specs.append({
                        "perturbation_type": 'none',
                        "perturbation_x": 0.0,
                        "perturbation_y": 0.0,
                        "star_perturbed": 0,
                        'rotation': 0,
                        "asterism": asterism,
                        'noise': 'position',
                        "asterism_w_position_uncertainty": asterisms.jitter_asterism(asterism, cfg.position_uncertainty, cfg.single_uncertainty),
                    })

            if cfg.measurement_n > 0:
                for _ in range(cfg.measurement_n):
                    asterism_specs.append({
                        "perturbation_type": 'none',
                        "perturbation_x": 0.0,
                        "perturbation_y": 0.0,
                        "star_perturbed": 0,
                        'rotation': 0,
                        "asterism": asterism,
                        'noise': 'measurement',
                    })

            for ast_spec in asterism_specs:
                for atm in atm_gens:
                    ap = atm.aperture_size
                    mask = tomo_cfg['mask'][ap]
                    target = beacon.Beacon(0, 0, 10, aperture_mask=mask)
                    
                    for atm_id, screens in enumerate(atm):
                        ast = ast_spec['asterism']
                        if ast_spec['noise'] == 'position':
                            ast_pu = ast_spec['asterism_w_position_uncertainty']
                        else:
                            ast_pu = ast

                        target_coeffs = target.cutout_coeffs(cfg.Z, cfg.D, cfg.alts, screens, atm.metapupil_diameters)
                        target_screen = target.total_cutout(cfg.D, cfg.alts, screens, atm.metapupil_diameters)
                        ast_coeffs = get_coeffs_for_asterism(ast_pu, cfg.Z, cfg.D, screens, cfg.alts, atm.metapupil_diameters)

                        if ast_spec['noise'] == 'measurement':
                            for i in range(len(ast_coeffs)):
                                ast_coeffs[i, 3:] = ast_coeffs[i, 3:] + np.random.normal(scale=C_mat)

                        for tomo in cfg.tomo_methods:

                            data_point = {
                                'group_id': group_id,
                                'asterism_id': asterism_id,
                                'atmosphere_id': atm_id,
                                'asterism_type': asterism_type,
                                "perturbation_type": ast_spec["perturbation_type"],
                                "perturbation_x": ast_spec["perturbation_x"],
                                "perturbation_y": ast_spec["perturbation_y"],
                                "perturbation_star": ast_spec["star_perturbed"],
                                'perturbation_rotation': ast_spec['rotation'],
                                'noise': ast_spec['noise'],
                                'aperture_pixels': ap,
                                'tomo_method': tomo,
                                'noise': ast_spec['noise']
                            }

                            for sid, star in enumerate(ast):
                                data_point[f'star_{sid}'] = star.to_tuple()

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
    manager = mp.Manager()
    asterism_queue = manager.Queue()
    result_queue = manager.Queue()
    # all_asterisms = []

    print(f"Generating {cfg.n_asterisms} asterisms")
    groups = asterisms.get_base_asterisms(cfg.n_asterisms)
    for i, asterism_group in tqdm(enumerate(groups)):
        asts = asterisms.get_asterisms_from_group(asterism_group)
        for j, (ast_type, ast) in enumerate(asts.items()):
            if (cfg.augmented_asterisms) or ("raw" in ast_type):
                asterism_queue.put((i, j, ast_type, ast))
            
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

def bulk_compile_asterism_data(dest_filename):
    files = glob.glob(os.path.join(DATA_FOLDER, 'batch', '*.parquet'))
    if len(files) > 0:
        for noise_type in ['measurement', 'position', 'none']:
            df = pd.DataFrame()
            for f in files:
                df0 = pd.read_parquet(f)
                df1 = df0[df0['noise'] == noise_type]
                df = pd.concat([df, df1], ignore_index=True)

            file_path = os.path.join(DATA_FOLDER, dest_filename + f'_{noise_type}_noise.parquet')
            df.to_parquet(file_path, index=False)
        
        print(f"Merged data in {os.path.join(DATA_FOLDER, 'batch')} into {file_path}")

        # Save csv for checking values
        if False:
            file_path = os.path.join(DATA_FOLDER, dest_filename + '.csv')
            df.to_csv(file_path, index=False)
            print(f"Created csv file for checking in {file_path}")

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

    # Build dataframe from asterism quality data
    df = pd.read_parquet(os.path.join(DATA_FOLDER, 'asterism_quality_data.parquet'))
    # ast_df = pd.read_parquet(os.path.join(DATA_FOLDER, 'asterism_geometry_data.parquet'))

    for i in range(3):
        df[f'star_{i}'] = df[f'star_{i}'].apply(tuple)

    def compute_asterism_features(row):
        a = [beacon.Beacon.from_tuple(row[f'star_{i}']) for i in range(3)]
        # b = [beacon.Beacon.from_tuple(row[f'star_{i}_baseline']) for i in range(3)]

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
            'f_distance_to_closest_edge' : asterisms.distance_to_closest_line(a),
            'f_perimeter' : asterisms.perimeter(a),
            'f_min_interior_angle' : asterisms.min_angle(a),
            'f_maximum_interior_angle' : asterisms.max_angle(a),
            'f_gaussian_area_within_triangle' : asterisms.gaussian_area_within_triangle(a),
            'f_overlap' : asterisms.overlap(a, cfg.alts),
            'f_target_in_triangle' : asterisms.is_target_in_triangle(a),
            # 'f_circular_similarity': asterisms.circular_similarity(a, b),
        }

        for i in range(3):
            data[f'star_{i}'] = row[f'star_{i}']

        return data
    print(df.shape[0])

    if cfg.perturbation_analysis == True:
        perturbation_analysis(df)

    methods_to_compare = ['glao', 'lsq_T', 'lsq_K']

    df = df[(df['perturbation_type'] == 'none')]
    perturbation_columns = df.columns[df.columns.str.contains('perturbation_')]
    df = df.drop(columns=perturbation_columns)
    df = df.drop(columns=['aperture_pixels'])

    result_columns = ['recon_rmse', 'coeff_rmse']
    df = df.drop(columns='strehl_ratio')

    # Calculate performance relative to baseline (open loop) performance
    baseline_factors = ['group_id', 'asterism_id', 'atmosphere_id', 'asterism_type']
    baseline = df[df['tomo_method'] == 'open_loop'].copy()
    baseline = baseline.rename(columns={col: f"{col}_baseline" for col in result_columns})
    baseline = baseline.drop(columns=['tomo_method', 'star_0', 'star_1', 'star_2'])  # to allow joining on all ID columns
    df = df.merge(baseline, on=baseline_factors)

    for col in result_columns:
        df[f'{col}_rel'] = df[col] / df[f'{col}_baseline']
        df[f'{col}_rel_log'] = (df[col] / df[f'{col}_baseline']).apply(np.log)

    df = df[df['tomo_method'].isin(methods_to_compare)]

    # Statistical analysis

    # Aggregate results along the atmosphere column.
    result_col_rel = [f'{col}_rel{t}' for col in result_columns for t in ['', '_log']]

    id_factors = ['group_id', 'asterism_id', 'tomo_method', 'asterism_type', 'star_0', 'star_1', 'star_2']

    #TODO: most of these are not normal, so need to do median and iqr instead.
    agg_dict = {
        col: ['mean', 'std', 'median'] for col in result_col_rel + result_columns
        # col: ['mean', 'std', 'sem', shapiro_p] for col in result_col_rel + result_columns
    }

    df = df.groupby(id_factors)[result_col_rel + result_columns].agg(agg_dict).reset_index()
    df.columns = ['_'.join(col).rstrip('_') if isinstance(col, tuple) else col for col in df.columns]

    print(df.shape[0])

    # Compute baseline asterism - the minimum equilateral asterism from the given stars (NO LONGER NEEDED)
    ast_df = df[['group_id', 'asterism_id', 'asterism_type','star_0', 'star_1', 'star_2']].copy()
    ast_df = ast_df.drop_duplicates(subset=['star_0', 'star_1', 'star_2'])
    star_cols = [f'star_{i}' for i in range(3)]
    # baseline_asts = ast_df[ast_df['asterism_type'] == 'star_0_circular'].copy()
    # baseline_asts = baseline_asts.rename(columns={col: f'{col}_baseline' for col in star_cols})
    # baseline_asts = baseline_asts.drop(columns=['asterism_id', 'asterism_type'])
    # ast_df = ast_df.merge(baseline_asts, on=['group_id'])

    # Compute features from asterism data
    df0 = ast_df.apply(compute_asterism_features, axis=1)
    df0 = pd.DataFrame(df0.tolist(), index=df0.index)
    df = df.merge(df0, on=star_cols, how='inner')

    # Group asterism types
    asterism_group = {f'star_{i}_circular': 'Single Star Circular' for i in range(5)}
    for i in range(10):
        asterism_group[f'sg{i}_raw'] = f'Raw'
        asterism_group[f'sg{i}_equal_angle'] = f'Equal Angle'
        asterism_group[f'sg{i}_mean_dist'] = f'Mean Distance'
        asterism_group[f'sg{i}_mean_circular'] = f'Mean Circular'

    df['Asterism Group'] = df['asterism_type'].map(asterism_group)

    df.to_parquet(os.path.join(DATA_FOLDER, 'asterism_quality_data_mean.parquet'), index=False)

def display_results():

    df = pd.read_parquet(os.path.join(DATA_FOLDER, 'asterism_quality_data_mean.parquet'))

    feature_columns = ['f_distance_0', 'f_distance_1', 'f_distance_2',
            'f_angle_0', 'f_angle_1', 'f_angle_2',
            'f_mean_distance', 
            'f_min_distance', 
            'f_max_distance',
            'f_area', 
            'f_centroid_distance', 
            'f_circumcentre_distance',
            'f_circumcircle_radius', 
            # 'f_length_ratio', 
            # 'f_average_side_length',
            # 'f_minimum_side_length', 
            # 'f_maximum_side_length', 
            'f_incentre_distance',
            'f_incircle_radius', 
            # 'f_perimeter_area_ratio', 
            # 'f_closest_furthest_side_length',
            'f_distance_to_closest_edge', 
            # 'f_perimeter', 
            # 'f_min_interior_angle',
            # 'f_maximum_interior_angle', 
            # 'f_gaussian_area_within_triangle', 
            'f_overlap',
            'f_target_in_triangle', 
            # 'f_circular_similarity'
            ]
    n_features = len(feature_columns)

    # plot_features(df, feature_columns)
    # comparison(df, feature_columns)
    test_column = 'recon_rmse_rel_log_mean'
    ast_type_list = ['_', 'raw']

    df = df[(df['tomo_method'] == 'lsq_T')] # & df['asterism_type'].str.contains('raw')]

    # dimensionality_reduction(df, feature_columns, y_col=test_column)
    # dimensionality_reduction_ML(df, feature_columns, y_col=test_column)

    # data_correlation(df)

    regression_results = []

    # for n in range(n_features - 1, n_features + 1):
    #     print(f"Running combinations of size {n}")
    #     for combination in combinations(feature_columns, n):
    regression_results += run_regression_models(df, feature_columns, y_col=test_column)
    regression_results.append(run_ML_regression_models(df, feature_columns, y_col=test_column))

    dfr = pd.DataFrame(regression_results)

    print("R² Results")
    r2_results = dfr.nlargest(3, 'R²')
    for _, row in r2_results.iterrows():
        print(f"R²: {row['R²']}; MSE: {row['MSE']}; Features: {row['Features']}")

    # print("MSE Results")
    # mse_results = dfr.nsmallest(10, 'MSE')
    # for _, row in mse_results.iterrows():
    #     print(f"MSE: {row['MSE']}; R²: {row['R²']}; Features: {row['Features']}")

    return # END Filter.filter_asterisms

def perturbation_analysis(df):
    tomo_methods = ['glao', 'lsq_T', 'lsq_K']
    metrics = ['coeff_rmse', 'recon_rmse']

    df_p = df.copy()
    df_s = df.copy()
    df_r = df.copy()
    
    # Dropping duplicates is very important here - this ensures a single mapping, not multiple between each id and type
    # df_p = df_p.merge(ast_df[['asterism_id', 'asterism_type']].drop_duplicates(), on='asterism_id', how='inner')

    df_p = df_p[df_p['perturbation_type'] == 'full']
    df_p = df_p.drop(columns=['perturbation_type', 'perturbation_star', 'aperture_pixels', 'perturbation_rotation'])
    perturbed_methods = ['lsq_T', 'lsq_K', 'glao']
    df_p = df_p[df_p['tomo_method'].isin(perturbed_methods)]

    df_s = df_s[df_s['perturbation_type'] == 'single']
    df_s['perturbation_star'] = df_s['perturbation_star'].fillna(0)
    df_s = df_s.drop(columns=['perturbation_type', 'aperture_pixels', 'perturbation_rotation'])
    df_s = df_s[df_s['tomo_method'].isin(perturbed_methods)]

    df_r = df_r[df_r['perturbation_type'] == 'rotate']
    df_r = df_r.drop(columns=['perturbation_type', 'perturbation_star', 'perturbation_x', 'perturbation_y'])
    df_r = df_r[df_r['tomo_method'].isin(perturbed_methods)]

    plots = [(method, metric) for method in tomo_methods for metric in metrics]

    # errors = {}
    for metric in metrics:
        baseline = df_p[(df_p['perturbation_x'] == 0.0) & (df_p['perturbation_y'] == 0.0)]
        baseline = baseline[['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id', metric]]
        baseline = baseline.rename(columns={metric: f'{metric}_baseline'})
        df0 = df_p.merge(baseline, on=['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id'], how='left')
        df_p[f'{metric}_diff'] = df0[metric] - df0[f'{metric}_baseline']

        baseline = df_s[(df_s['perturbation_x'] == 0.0) & (df_s['perturbation_y'] == 0.0)]
        baseline = baseline[['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id', 'perturbation_star', metric]]
        baseline = baseline.rename(columns={metric: f'{metric}_baseline'})
        df0 = df_s.merge(baseline, on=['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id', 'perturbation_star'], how='left')
        df_s[f'{metric}_diff'] = df0[metric] - df0[f'{metric}_baseline']

        baseline = df_r[(df_r['perturbation_rotation'] == 0.0)]
        baseline = baseline[['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id', metric]]
        baseline = baseline.rename(columns={metric: f'{metric}_baseline'})
        df0 = df_r.merge(baseline, on=['group_id', 'asterism_id', 'tomo_method', 'atmosphere_id'], how='left')
        df_r[f'{metric}_diff'] = df0[metric] - df0[f'{metric}_baseline']


    df0 = df_p.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation_x', 'perturbation_y'])[[metric for metric in metrics]].mean().reset_index()
    df_p = df_p.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation_x', 'perturbation_y'])[[f'{metric}_diff' for metric in metrics]].mean().reset_index()
    df_p = df_p.merge(df0, on=['group_id', 'asterism_id', 'tomo_method', 'perturbation_x', 'perturbation_y'])

    df0 = df_s.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation_x', 'perturbation_y'])[[metric for metric in metrics]].mean().reset_index()
    df_s = df_s.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation_x', 'perturbation_y'])[[f'{metric}_diff' for metric in metrics]].mean().reset_index()
    df_s = df_s.merge(df0, on=['group_id', 'asterism_id', 'tomo_method', 'perturbation_x', 'perturbation_y'])

    df0 = df_r.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation_rotation'])[[metric for metric in metrics]].mean().reset_index()
    df_r = df_r.groupby(['group_id', 'asterism_id', 'tomo_method', 'perturbation_rotation'])[[f'{metric}_diff' for metric in metrics]].mean().reset_index()
    df_r = df_r.merge(df0, on=['group_id', 'asterism_id', 'tomo_method', 'perturbation_rotation'])

    for method, metric in plots:  

        n_x = len(df_p['perturbation_x'].unique())
        n_y = len(df_p['perturbation_y'].unique())

        if n_x > 0 and n_y > 0:
            fig, axs = plt.subplots(n_y, n_x)
            f2, axs2 = plt.subplots(n_y, n_x)
            fig.suptitle(f'Perturbation of {method}: {metric}')
            f2.suptitle(f'Perturbation of {method}: {metric} (Outliers Removed)')

            for i_x, p_x in enumerate(sorted(df_p['perturbation_x'].unique())):
                for i_y, p_y in enumerate(sorted(df_p['perturbation_y'].unique())):
                    if p_x != 0.0 or p_y != 0.0:
                        errors = df_p[(df_p['perturbation_x'] == p_x) & (df_p['perturbation_y'] == p_y) & (df_p['tomo_method'] == method)][f"{metric}_diff"].tolist()
                        errors = np.array(errors)
                        outlier_mask = util.is_outlier(errors)
                        outliers = errors[outlier_mask]
                        errors_no_outliers = list(errors[~outlier_mask])
                        print(f"{method}, {metric}, ({p_x}, {p_y}) has {len(outliers)} outliers/{len(outlier_mask)} values")
                        print(f"{method}, {metric}, ({p_x}, {p_y}) has mean: {np.mean(errors)}, variance: {np.var(errors)}")

                        axs[i_y, i_x].hist(errors, bins=30, color='g')
                        axs2[i_y, i_x].hist(errors_no_outliers, bins=30, color='g')
                        
                    else:
                        values = df_p[(df_p['perturbation_x'] == p_x) & (df_p['perturbation_y'] == p_y) & (df_p['tomo_method'] == method)][metric].tolist()
                        axs[i_y, i_x].hist(values, bins=30, color='r')
                        axs2[i_y, i_x].hist(values, bins=30, color='r')
                
            for i in range(len(df_p['perturbation_x'].unique())):
                axs[n_y - 1, i].set_xlabel('RMSE (rads)')
                axs2[n_y - 1, i].set_xlabel('RMSE (rads)')
                        
            for i in range(len(df_p['perturbation_y'].unique())):
                axs[i, 0].set_ylabel('Count')
                axs2[i, 0].set_ylabel('Count')
            # Column labels
            for j, x in enumerate(sorted(df_p['perturbation_x'].unique())):
                label = f'{x:.1f}"'
                fig.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)
                f2.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)

            # Row labels
            for i, y in enumerate(sorted(df_p['perturbation_y'].unique())):
                label = f'{y:.1f}"'
                fig.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)
                f2.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)

        ns_y = len(df_s['perturbation_y'].unique())
        ns_x = len(df_s['perturbation_x'].unique())

        if ns_x > 0 and ns_y > 0:
            sfig, saxs = plt.subplots(ns_y, ns_x)
            sfig.suptitle(f'Single star perturbation of {method}: {metric}')

            for i_x, p_x in enumerate(sorted(df_s['perturbation_x'].unique())):
                for i_y, p_y in enumerate(sorted(df_s['perturbation_y'].unique())):
                    if p_x != 0.0 or p_y != 0.0:
                        errors = df_s[(df_s['perturbation_x'] == p_x) & (df_s['perturbation_y'] == p_y) & (df_s['tomo_method'] == method)][f"{metric}_diff"].tolist()

                        saxs[i_y, i_x].hist(errors, bins=10, color='b')
                    else:
                        values = df_s[(df_s['perturbation_x'] == p_x) & (df_s['perturbation_y'] == p_y) & (df_s['tomo_method'] == method)][metric].tolist()
                        saxs[i_y, i_x].hist(values, bins=10, color='r')

            for i in range(3):
                saxs[2, i].set_xlabel('RMSE (rads)')
                saxs[i, 0].set_ylabel('Count')

            for j, x in enumerate(sorted(df_s['perturbation_x'].unique())):
                label = f'{x:.1f}"'
                sfig.text(0.19 + j * 0.16, 0.92, label, ha='center', va='center', fontsize=12)

            for i, y in enumerate(sorted(df_s['perturbation_y'].unique())):
                label = f'{y:.1f}"'
                sfig.text(0.08, 0.83 - i * 0.16, label, ha='center', va='center', rotation='vertical', fontsize=12)


        n_r = len(df_r['perturbation_rotation'].unique())

        if n_r > 0:
            f_r, axsr = plt.subplots(1, n_r)
            f_r.suptitle(f'Rotation of {method}: {metric}')
            for i_r, p_r in enumerate(sorted(df_r['perturbation_rotation'].unique())):
                if p_r != 0.0:
                    errors = df_r[(df_r['perturbation_rotation'] == p_r) & (df_r['tomo_method'] == method)][f"{metric}_diff"].tolist()
                    axsr[i_r].hist(errors, bins=30)
                else:
                    values = df_r[(df_r['perturbation_rotation'] == p_r) & (df_r['tomo_method'] == method)][metric].tolist()
                    axsr[i_r].hist(values, bins=30, color='r')
                axsr[i_r].set_xlabel('RMSE (rads)')
                axsr[i_r].set_title(f'Rotation = {p_r:.2f}rad')
            axsr[0].set_ylabel('Count')


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
    
    for Y in df.columns[df.columns.str.contains('coeff') & df.columns.str.contains('mean')]:
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

            # g = sns.lmplot(df, x=feature, y=Y, hue='Asterism Group', col='tomo_method')
            # g.figure.suptitle(f"Output Type: \"{Y}\", Feature: \"{feature}\". All asterism groups separate")
            # g = sns.lmplot(df, x=feature, y=Y, col='tomo_method')
            # g.figure.suptitle(f"Output Type: \"{Y}\", Feature: \"{feature}\". All asterism groups combined")
            g = sns.lmplot(df[df["Asterism Group"] == 'Raw'], x=feature, y=Y, col='tomo_method')
            g.figure.suptitle(f"Output Type: \"{Y}\", Feature: \"{feature}\". No augmented asterisms")

def dimensionality_reduction(df, feature_list, y_col):
    # Data for regression testing
    X = df[feature_list]
    y = df[y_col]

    print("Calculating 2D UMAP")
    plt.figure()
    jitter_strength = 0.01
    Xj = X + np.random.randn(*X.shape) * (X.std().values * jitter_strength)
    # Xj = X + np.random.normal(loc=0, scale=0.05, size=X.shape)
    umap_reducer = umap.UMAP(n_components=2, n_neighbors=15)
    X_umap = umap_reducer.fit_transform(Xj)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
    plt.title("UMAP 2D")

    # print("Calculating 3D UMAP")
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # umap_reducer = umap.UMAP(n_components=3)
    # X_umap = umap_reducer.fit_transform(X)
    # ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=y, cmap='viridis')
    # fig.suptitle("UMAP 3D")

    print("Calculating 2D t-SNE")
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    plt.title("t-SNE 2D")

    # print("Calculating 3D t-SNE")
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # tsne2 = TSNE(n_components=3)
    # X_tsne2 = tsne2.fit_transform(X)
    # ax.scatter(X_tsne2[:, 0], X_tsne2[:, 1], X_tsne2[:, 2], c=y, cmap='viridis')
    # fig.suptitle("t-SNE 3D")

    # Display higher dimensional data as 2D using Principal Component Analysis
    print("Calculating 2D PCA")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title("PCA 2D")

    # print("Calculating 3D PCA")
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # PCA_model = PCA(n_components=3).fit(X)
    # X_reduced = PCA_model.transform(X)
    # ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis')
    # fig.suptitle("PCA 3D")

    n_components = 10
    if n_components < len(feature_list):
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
    
def data_correlation(df):
    # Show correlations between variables
    result_columns = list(df.columns[df.columns.str.contains('mean') & df.columns.str.contains('recon')])

    df0 = df.filter(regex='^f_.*')
    feature_columns = list(df0.columns)

    plt.figure()

    correlations = df[result_columns + feature_columns].corr().loc[feature_columns, result_columns]
    sns.heatmap(correlations, annot=True)

    return # END Filter.data_correlation

def run_regression_models(df, feature_columns, y_col, split_size=0.2):
    data = []
    # Data for regession testing
    X = df[feature_columns]
    y = df[y_col]
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=split_size)

    # Partial Least Squares Regression (PLS)
    model = PLSRegression(n_components=2)
    data.append(regression_data(X_trn, X_tst, y_trn, y_tst, model, f"Partial Least Squares Regression"))

    # Principal Component Regression (PCR)
    pcr = make_pipeline(PCA(n_components=2), LinearRegression())
    data.append(regression_data(X_trn, X_tst, y_trn, y_tst, pcr, "Principal Component Regression"))

    # Ridge Regression
    rdg = Ridge(alpha=0.1)
    data.append(regression_data(X_trn, X_tst, y_trn, y_tst, rdg, f"Ridge Regression"))
    
    # Lasso Regression
    lso = Lasso(alpha=0.1)
    data.append(regression_data(X_trn, X_tst, y_trn, y_tst, lso, "Lasso Regression"))

    for kernel in ['linear', 'poly', 'rbf']: #, 'sigmoid']:
        svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
        regression(X_trn, X_tst, y_trn, y_tst, svr, f"Support Vector Regression [{kernel}]")

    return data # END Filter.run_regression_models

def run_ML_regression_models(df, feature_columns, y_col, split_size=0.2):
    # Data for regession testing
    X = df[feature_columns]
    y = df[y_col]
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=split_size)

    n_features = len(feature_columns)

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
    return regression_data(X_trn, X_tst, y_trn, y_tst, reg_model, f"ML Regression")
    # regression(X_trn, X_tst, y_trn, y_tst, reg_model, f"ML Regression")

    # reg_model.fit(X_trn, y_trn)

    # y_pred = reg_model.predict(X_tst).flatten()

    # df0 = pd.DataFrame()
    # df0['prediction'] = y_pred
    # df0['test'] = y_tst.to_numpy()

    # for feature in feature_columns[6:]:
    #     plt.figure()
    #     plt.scatter(df0['test'], df0['prediction'], c=X_tst[feature], cmap='viridis')
    #     plt.plot([df0['test'].min(), df0['test'].max()], [df0['test'].min(), df0['test'].max()], 'r:')
    #     plt.xlabel('Test Value')
    #     plt.ylabel('Prediction')
    #     plt.title(f"Regression highlighting feature: {feature} [{ast_type}]")

    return # END Filter.run_ML_regression

def dimensionality_reduction_ML(df, feature_columns, y_col, split_size=0.2):
    X = df[feature_columns]
    y = df[y_col]
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=split_size)

    n_features = len(feature_columns)

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

    return

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
    N = 10
    A = 3

    def plot_circ():
        theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
        rho = np.ones_like(theta) * 120
        x, y = util.pol2cart(rho, theta)
        plt.plot(x, y, "k:", alpha=0.2)
    
    # Line Through Target
    def get_asterism_line_through_target():
        theta = np.random.uniform(0, 2*np.pi)
        return [beacon.Beacon(theta + ((i % 2) * np.pi), gamma=np.random.uniform(0, 120)) for i in range(A)]
    
    def get_asterism_line_off_target():
        x_start = np.random.uniform(-120, 120)
        y_start = np.random.uniform(-np.sqrt(120**2-x_start**2), np.sqrt(120**2-x_start**2))
        y_end = np.random.uniform(-120, 120)
        x_end = np.random.uniform(-np.sqrt(120**2-y_end**2), np.sqrt(120**2-y_end**2))
        scale = np.random.uniform(0.1, 0.9)
        x_mid = scale * (x_end - x_start) + x_start
        y_mid = scale * (y_end - y_start) + y_start
        b_start = beacon.Beacon(0, 0).offset(x_start, y_start)
        b_mid = beacon.Beacon(0, 0).offset(x_mid, y_mid)
        b_end = beacon.Beacon(0, 0).offset(x_end, y_end)

        return [b_start, b_mid, b_end]


    def get_asterism_circular():
        theta0 = np.random.uniform(0, 2*np.pi/3)
        gamma = np.random.uniform(5, 120)
        return [beacon.Beacon(theta0 + (i * 2 * np.pi / 3), gamma) for i in range(A)]

    def get_asterism_tight_cluster_far_off_target():
        max_g = 100
        gamma = np.random.uniform(45, max_g)
        theta = np.random.uniform(0, 2*np.pi)
        lim = np.sqrt(((120-max_g)**2)/2)
        offset_x = np.random.uniform(-lim, lim, size=3)
        offset_y = np.random.uniform(-lim, lim, size=3)
        base = beacon.Beacon(theta, gamma)

        return [base.offset(x=offset_x[i], y=offset_y[i]) for i in range(3)]

    def get_asterism_arc():
        t0 = np.random.uniform(0, 2*np.pi)
        gamma = np.random.uniform(5, 120)
        angle = np.random.uniform(np.pi/3, 3*np.pi/2)
        return [beacon.Beacon(t0+(angle*i/3), gamma) for i in range(3)]
    
    ast_funcs = {
        "Line Through Target": get_asterism_line_through_target,
        "Line Off Target": get_asterism_line_off_target,
        "Circular Asterism": get_asterism_circular,
        "Tight Cluster Far Off-Target": get_asterism_tight_cluster_far_off_target,
        "Arc": get_asterism_arc,
    }
    
    for name, func in ast_funcs.items():
        plt.figure()
        for _ in range(N):
            asterism = func()
            x = [b.position_arcsec()[0] for b in asterism]
            y = [b.position_arcsec()[1] for b in asterism]

            plt.plot(x, y, "+:")
        plt.suptitle(f"Asterism Type: {name}")
        plt.plot([0], [0], 'ko')
        plot_circ()
        ax = plt.gca()
        ax.set_aspect('equal')

    plt.show()

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

def regression_data(X_train, X_test, y_train, y_test, model, name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).flatten()

    reg_data = {
        'model': name,
        'R²': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'Features': X_train.columns
    }

    return reg_data

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
                outputs = self.model(batch_X.to(self.device)).squeeze()
                loss = self.criterion(outputs, batch_y.to(self.device).squeeze())
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
    
    
