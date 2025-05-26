import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
import time
import traceback

import pandas as pd
import seaborn as sns

from tomography import asterisms, beacon, phase_screen
from tomography.tomography import GLAO, LeastSquares, get_coeffs_for_asterism, AO
from basis import zernike
import util
import joblib

def run_tests():
    f = Filter()

    # f.run_asterism_comparison()
    f.filter_asterisms()

    load_angle_data_test()
    
    plt.show()

def load_angle_data_test():
    with open(f'data/satellites/angle_data_mag6.pak', 'rb') as f:
        data = joblib.load(f)

    print(data[:5])

    with open(f'data/satellites/angle_data_filtered.pak', 'rb') as f:
        data = joblib.load(f)

    print(data[:5])

class Filter():
    def __init__(self):
        self.N = 21
        self.Z = zernike.Zernike(self.N)
        self.D = 1.8
        self.r0 = 0.07
        self.ap_pix = 100
        self.alts = [1000, 5000, 10000]
        self.n_asterisms = 1000
        self.glao = GLAO(self.Z, self.D, self.N, self.alts, self.ap_pix)
        self.ao = AO(self.Z, self.D, self.N, self.alts, self.ap_pix)
        self.lsq = LeastSquares(self.Z, self.D, self.N, self.alts, self.ap_pix, r0=self.r0)
        self.target = beacon.Beacon(0, 0, 10)
        self.total = -1
        self.n_atm = 512

    def worker(self, atm, aq, rq, l):
        try:
            tstart = time.time()
            id = os.getpid()
            l.acquire()
            while not aq.empty():
                i, asterism = aq.get()
                l.release()
                t0 = time.time()
                
                glao_mse = []
                tiptilt_mse = []
                lsq_mse = []
                ol_mse = []
                ao_mse = []
                for screens in atm:
                    target_coeffs = self.target.cutout_coeffs(self.Z, self.D, self.alts, screens, atm.metapupil_diameters)
                    ast_coeffs = get_coeffs_for_asterism(asterism, self.Z, self.D, screens, self.alts, atm.metapupil_diameters)
                    tiptilt_coeffs = np.zeros_like(target_coeffs)
                    tiptilt_coeffs[:3] = target_coeffs[:3]
                    glao_coeffs = self.glao.run(asterism, self.target, ast_coeffs)
                    lsq_coeffs = self.lsq.run(asterism, self.target, ast_coeffs)
                    ao_coeffs = self.ao.run(asterism, self.target, ast_coeffs)
                    ol_coeffs = np.zeros_like(lsq_coeffs)
                    glao_mse.append(util.rmse(target_coeffs, glao_coeffs))
                    tiptilt_mse.append(util.rmse(target_coeffs, tiptilt_coeffs))
                    ao_mse.append(util.rmse(target_coeffs, ao_coeffs))
                    lsq_mse.append(util.rmse(target_coeffs, lsq_coeffs))
                    ol_mse.append(util.rmse(target_coeffs, ol_coeffs))
                
                glao_mse = np.mean(glao_mse)
                tiptilt_mse = np.mean(tiptilt_mse)
                lsq_mse = np.mean(lsq_mse)
                ol_mse = np.mean(ol_mse)
                ao_mse = np.mean(ao_mse)
                rq.put((i, ol_mse, glao_mse, lsq_mse, tiptilt_mse, ao_mse))

                t1 = time.time()
                dt = t1 - t0
                dt_total = t1 - tstart 
                print(f"[{id:5}] Simulated asterism {i+1}/{self.total}. Took {dt:5.2f}s. Total time since start: {dt_total:.2f}s")
                l.acquire()
            l.release()
        except Exception as e:
            print(type(e))    # the exception type
            print(e.args)     # arguments stored in .args
            print(e)
            print(traceback.format_exc(), flush=True)
        return

    def run_asterism_comparison(self):

        all_asterisms = asterisms.generate_variable_asterisms(n=self.n_asterisms, zero_rate=0.0, fixed_rate=0.0)
        # fixed_asterisms = asterisms.generate_fixed_asterisms(10, 5, theta_0_random=True)
        # for angle in range(10, 120, 5):
        #     fixed_asterisms += asterisms.generate_fixed_asterisms(10, angle, theta_0_random=True)
        # all_asterisms += fixed_asterisms
        self.total = len(all_asterisms)


        print(f"Generating {self.n_atm} simulated atmospheres")
        atm = phase_screen.Atmosphere(self.n_atm, self.alts, r0=self.r0, D=self.D, aperture_size=self.ap_pix)
        print(f"Completed simulating {self.n_atm} atmospheres")

        manager = Manager()
        asterism_queue = manager.Queue()
        result_queue = manager.Queue()
        lock = manager.Lock()

        for i, asterism in tqdm(enumerate(all_asterisms)):
            asterism_queue.put((i, asterism))

        n_processes = 8
        print("Run multi-processing for asterism quality filtering")

        with Pool(processes=n_processes) as p:
            workers = [p.apply_async(self.worker, args=(atm, asterism_queue, result_queue, lock)) for _ in range(n_processes)]

            res = [p.get() for p in workers]

        print("Multi-processing completed")
        # good_asterisms = []
        # ok_asterisms = []
        # bad_asterisms = []

        # x_good = []
        # y_good = []

        # x_ok = []
        # y_ok = []

        # x_bad = []
        # y_bad = []

        results = []

        while not result_queue.empty():
            j, ol_mse, glao_mse, lsq_mse, tiptilt_mse, ao_mse = result_queue.get()

            if j % 100 == 0:
                print(f"Loading results for asterism {j+1}/{self.total}")
            asterism = all_asterisms[j]

            # i = np.argmin([x.gamma for x in asterism])
            # offset = asterism[i].theta
            # ys = []
            # xs = []
            # for b in asterism:
            #     b1 = beacon.Beacon(b.theta, b.gamma, b.magnitude, arcseconds=False)
            #     b1.theta = b1.theta - offset
            #     x, y = util.rad2arcsec(b1.position(1))
            #     xs.append(x)
            #     ys.append(y)

            results.append({
                'asterism_id':  j,
                'gs_1':         asterism[0].to_simple_str(),
                'gs_2':         asterism[1].to_simple_str(),
                'gs_3':         asterism[2].to_simple_str(),
                'ol_mse':       ol_mse,
                'lsq_mse':      lsq_mse,
                'glao_mse':     glao_mse,
                'ao_mse':       ao_mse,
                'tiptilt_mse':  tiptilt_mse,
            })

            # if lsq_mse < glao_mse:
            # # if lsq_mse < tiptilt_mse:
            #     good_asterisms.append(asterism)
            # elif lsq_mse < ol_mse:
            #     ok_asterisms.append(asterism)
            # else:
            #     bad_asterisms.append(asterism)
        

            # xs.append(xs[0])
            # ys.append(ys[0])

            # if lsq_mse < glao_mse:
            # # if lsq_mse < tiptilt_mse:
            #     x_good.append(xs)
            #     y_good.append(ys)
            # elif lsq_mse < ol_mse:
            #     x_ok.append(xs)
            #     y_ok.append(ys)
            # else:
            #     x_bad.append(xs)
            #     y_bad.append(ys)

        df = pd.DataFrame(results)
        df.to_csv('data/satellites/asterism_quality_data.csv')

    def filter_asterisms(self):
        df = pd.read_csv('data/satellites/asterism_quality_data.csv', index_col="asterism_id")

        asterism_list = [[beacon.Beacon.from_simple_str(df.loc[idx, f'gs_{i+1}']) for i in range(3)] for idx in range(df.shape[0])]
        min_distance = [asterisms.min_distance(a) for a in asterism_list]
        ave_distance = [asterisms.ave_distance(a) for a in asterism_list]
        centroid_distance = [asterisms.centroid_distance(a) for a in asterism_list]
        length_ratio = [asterisms.length_ratio(a) for a in asterism_list]
        overlap_full = [asterisms.overlap(a, [1000, 5000, 10000]) for a in asterism_list]
        # overlap_1st = [asterisms.overlap(a, [1000]) for a in asterism_list]
        # overlap_2nd = [asterisms.overlap(a, [5000]) for a in asterism_list]
        # overlap_3rd = [asterisms.overlap(a, [10000]) for a in asterism_list]

        # Use Z-score normalization on difference of lsq rmse to open loop rmse. Larger values here imply better performance
        performance = df['ol_mse'] - df['lsq_mse']
        performance = (performance - performance.mean()) / performance.std()

        df0 = pd.DataFrame()

        df0['performance'] = performance
        df0['min_distance'] = min_distance
        df0['average_distance'] = ave_distance
        df0['centroid_distance'] = centroid_distance
        df0['length_ratio'] = length_ratio
        df0['overlap_full'] = overlap_full
        # df0['overlap_1st'] = overlap_1st
        # df0['overlap_2nd'] = overlap_2nd
        # df0['overlap_3rd'] = overlap_3rd

        # df1 = pd.DataFrame()
        # df1['LSQ MSE'] = performance
        # for lbl in ['min_distance', 'average_distance', 'centroid_distance']:
        #     df1[lbl] = 1 - (df0[lbl] - df0[lbl].min()) / (df0[lbl].max() - df0[lbl].min())
            # df1[lbl] = (df0[lbl] - df0[lbl].mean()) / (df0[lbl].std())
        # for lbl in ['length_ratio', 'overlap_full']:
        #     df1[lbl] = (df0[lbl] - df0[lbl].min()) / (df0[lbl].max() - df0[lbl].min())
            # df1[lbl] = (df0[lbl].mean() - df0[lbl]) / (df0[lbl].std())
        
        # df1['md*ad'] = df1['min_distance']      * df1['average_distance']
        # df1['md*cd'] = df1['min_distance']      * df1['centroid_distance']
        # df1['md*lr'] = df1['min_distance']      * df1['length_ratio']
        # df1['md*of'] = df1['min_distance']      * df1['overlap_full']
        # df1['ad*cd'] = df1['average_distance']  * df1['centroid_distance']
        # df1['ad*lr'] = df1['average_distance']  * df1['length_ratio']
        # df1['ad*of'] = df1['average_distance']  * df1['overlap_full']
        # df1['cd*lr'] = df1['centroid_distance'] * df1['length_ratio']
        # df1['cd*of'] = df1['centroid_distance'] * df1['overlap_full']
        # df1['lr*of'] = df1['length_ratio']      * df1['overlap_full']

        # df1['md*ad*cd'] = df1['min_distance']       * df1['average_distance']   * df1['centroid_distance']
        # df1['md*ad*lr'] = df1['min_distance']       * df1['average_distance']   * df1['length_ratio']
        # df1['md*ad*of'] = df1['min_distance']       * df1['average_distance']   * df1['overlap_full']
        # df1['md*cd*lr'] = df1['min_distance']       * df1['centroid_distance']  * df1['length_ratio']
        # df1['md*cd*of'] = df1['min_distance']       * df1['centroid_distance']  * df1['overlap_full']
        # df1['md*lr*of'] = df1['min_distance']       * df1['length_ratio']       * df1['overlap_full']
        # df1['ad*cd*lr'] = df1['average_distance']   * df1['centroid_distance']  * df1['length_ratio']
        # df1['ad*cd*of'] = df1['average_distance']   * df1['centroid_distance']  * df1['overlap_full']
        # df1['ad*lr*of'] = df1['average_distance']   * df1['length_ratio']       * df1['overlap_full']
        # df1['cd*lr*of'] = df1['centroid_distance']  * df1['length_ratio']       * df1['overlap_full']

        # df1['md*ad*cd*lr'] = df1['min_distance']        * df1['average_distance']   * df1['centroid_distance']  * df1['length_ratio']
        # df1['md*ad*cd*of'] = df1['min_distance']        * df1['average_distance']   * df1['centroid_distance']  * df1['overlap_full']
        # df1['md*ad*lr*of'] = df1['min_distance']        * df1['average_distance']   * df1['length_ratio']       * df1['overlap_full']
        # df1['md*cd*lr*of'] = df1['min_distance']        * df1['centroid_distance']  * df1['length_ratio']       * df1['overlap_full']
        # df1['ad*cd*lr*of'] = df1['average_distance']    * df1['centroid_distance']  * df1['length_ratio']       * df1['overlap_full']

        # df1['md*ad*cd*lr*of'] = df1['min_distance'] * df1['average_distance'] * df1['centroid_distance'] * df1['length_ratio'] * df1['overlap_full']

        # print(df.columns)
        # df0 = df.drop(['gs_1', 'gs_2', 'gs_3', 'ol_mse', 'glao_mse', 'ao_mse', 'tiptilt_mse', 'Unnamed: 0'], axis=1)

        sns.lmplot(data=df0, y='LSQ MSE', x='min_distance')
        sns.lmplot(data=df0, y='LSQ MSE', x='average_distance')
        sns.lmplot(data=df0, y='LSQ MSE', x='centroid_distance')
        sns.lmplot(data=df0, y='LSQ MSE', x='length_ratio')
        sns.lmplot(data=df0, y='LSQ MSE', x='overlap_full')

        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*ad')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*cd')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*lr')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='ad*cd')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='ad*lr')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='ad*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='cd*lr')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='cd*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='lr*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*ad*cd')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*ad*lr')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*ad*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*cd*lr')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*cd*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*lr*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='ad*cd*lr')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='ad*cd*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='ad*lr*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='cd*lr*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*ad*cd*lr')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*ad*cd*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*ad*lr*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*cd*lr*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='ad*cd*lr*of')
        # sns.lmplot(data=df1, y='LSQ MSE', x ='md*ad*cd*lr*of')

        # sns.pairplot(df0, hue="LSQ MSE", palette="hot_r")

        return

        plt.figure()

        plt.xlabel("Arcseconds")
        plt.ylabel("Arcseconds")
        axis = plt.gca()
        axis.set_aspect('equal')
        for x, y in zip(x_bad, y_bad):
            plt.plot(x, y, 'rs--', alpha=0.02)
        for x, y in zip(x_ok, y_ok):
            plt.plot(x, y, 'bo--', alpha=0.04)
        for x, y in zip(x_good, y_good):
            plt.plot(x, y, 'g+-', alpha=0.3)

        plt.plot([0, 0], [-120, 120], 'k:', alpha=0.3)
        plt.plot([-120, 120], [0, 0], 'k:', alpha=0.3)
        n_points = 100
        for angle in [15, 30, 45, 60, 75, 90, 105, 120]:
            g = [util.pol2cart(120,t/n_points*(2*np.pi)) for t in range(n_points + 1)]
            alpha = 0.3 if angle %30 == 0 else 0.2
            plt.plot([x for x,_ in g], [y for _,y in g], 'k:', alpha=alpha)

        fig, axs = plt.subplots(1, 7)

        good_angle = [asterisms.max_angle(a) for a in good_asterisms]
        ok_angle = [asterisms.max_angle(a) for a in ok_asterisms]
        bad_angle = [asterisms.max_angle(a) for a in bad_asterisms]

        good_min_angle = [asterisms.min_angle(a) for a in good_asterisms]
        ok_min_angle = [asterisms.min_angle(a) for a in ok_asterisms]
        bad_min_angle = [asterisms.min_angle(a) for a in bad_asterisms]

        good_centroid = [asterisms.centroid_distance(a) for a in good_asterisms]
        ok_centroid = [asterisms.centroid_distance(a) for a in ok_asterisms]
        bad_centroid = [asterisms.centroid_distance(a) for a in bad_asterisms]

        good_perim = [asterisms.perimeter_area_ratio(a) for a in good_asterisms]
        ok_perim = [asterisms.perimeter_area_ratio(a) for a in ok_asterisms]
        bad_perim = [asterisms.perimeter_area_ratio(a) for a in bad_asterisms]

        good_length = [asterisms.length_ratio(a) for a in good_asterisms]
        ok_length = [asterisms.length_ratio(a) for a in ok_asterisms]
        bad_length = [asterisms.length_ratio(a) for a in bad_asterisms]

        good_asterisms = [[(b._gamma_arcsec, b.theta, b.magnitude) for b in a] for a in good_asterisms]
        bad_asterisms = [[(b._gamma_arcsec, b.theta, b.magnitude) for b in a] for a in bad_asterisms]
        ok_asterisms = [[(b._gamma_arcsec, b.theta, b.magnitude) for b in a] for a in ok_asterisms]

        good_min = [np.min([b[0] for b in a]) for a in good_asterisms]
        ok_min = [np.min([b[0] for b in a]) for a in ok_asterisms]
        bad_min = [np.min([b[0] for b in a]) for a in bad_asterisms]

        good_mean = [np.mean([b[0] for b in a]) for a in good_asterisms]
        ok_mean = [np.mean([b[0] for b in a]) for a in ok_asterisms]
        bad_mean = [np.mean([b[0] for b in a]) for a in bad_asterisms]

        fns = [('Minimum angle', good_min, ok_min, bad_min), 
               ('Average angle', good_mean, ok_mean, bad_mean),
               ('Centroid Distance', good_centroid, ok_centroid, bad_centroid),
               ('Perimeter Area Ratio', good_perim, ok_perim, bad_perim),
               ('Length Ratio', good_length, ok_length, bad_length),
               ('Minimum Triangle Angle', good_min_angle, ok_min_angle, bad_min_angle),
               ('Maximum Triangle Angle', good_angle, ok_angle, bad_angle)]
        for ax, (t, g, o, b) in zip(axs, fns):
            ax.set_title(t)
            ax.set_ylabel("Density")

            ax.hist(g, 20, density=True, histtype='step', color='g', alpha=0.5, label="Good")
            ax.hist(o, 20, density=True, histtype='step', color='b', alpha=0.5, label="OK")
            ax.hist(b, 20, density=True, histtype='step', color='r', alpha=0.5, label="Bad")

            if t == 'Maximum Triangle Angle':
                ax.set_xlabel("Radians")
                ax.legend()
            elif t == 'Minimum Triangle Angle':
                ax.set_xlabel("Radians")
            elif t == 'Length Ratio' or t == 'Perimeter Area Ratio':
                ax.set_xlabel("Ratio")
            else:
                ax.set_xlabel("Arcseconds")
        
        with open(f'data/satellites/angle_data_filtered_good.pak', 'wb') as f:
            joblib.dump(good_asterisms, f)

        with open(f'data/satellites/angle_data_filtered_ok.pak', 'wb') as f:
            joblib.dump(ok_asterisms, f)

        print(f"Good: {len(good_asterisms)}; OK: {len(ok_asterisms)}; Bad: {len(bad_asterisms)}")
