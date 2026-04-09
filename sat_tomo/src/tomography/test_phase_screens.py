"""
Calculate r0 for a set of three phase screens at different altitudes.
I want to have two modes:
Equal turbulence, and a realistic approach where the ground layer has the most distortion, but stil lresults in the same total r0 across all three layers.

"""
import numpy as np
from tomography import phase_screen, beacon
from scipy import stats
from util import circle
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as stats

def run_tests():
    test_r0()

def test_r0():
    
    N = 1000
    N_r0 = 1

    ap_pix = 100

    D = 1.8
    alts = [1000, 5000, 10000]

    data = {}
    target = beacon.Beacon(0,0,0)
    mask = target.aperture_mask

    rng = np.random.default_rng()

    row = 1
    row_end = row + N
    row_start = row + 1

    with open("data/tests/r0.csv", 'w') as f:
        f.write(f"Layer 1 r0 (Set),Layer 2 r0 (Set),Layer 3 r0 (Set),Total r0 (Set),Layer 1 r0 (Est),Layer 2 r0 (Est),Layer 3 r0 (Est),Total r0 (Est),Sorted,Norm Dist\n")
        for r0 in rng.normal(0.4, 0.1, (N_r0, 3)):

            r0[r0 < 0] = 0
            r0[r0 > 1.0] = 1.0

            r0 = list(r0)
            # if type(r0) is not list:
            #     r0 = [r0*(len(alts)**(3/5)) for _ in alts]

            r0_layers = np.array(r0)  # in meters
            r0_total = (np.sum(r0_layers**(-5/3)))**(-3/5)

            print(f"{r0[0]*100:.2f},{r0[1]*100:.2f},{r0[2]*100:.2f},{r0_total*100:.2f}")

            data[r0_total] = []

            gen = phase_screen.Atmosphere(n=N,
                                    layer_altitudes=alts,
                                    r0=r0,
                                    D=D,
                                    aperture_size=ap_pix)
            
            for atm in gen:
                row += 1
                if row > row_end:
                    row_start = row
                    row_end += N
                # n_atm = []
                # for scr, er0 in zip(atm, r0):
                #     n_atm.append(np.sqrt(np.var(scr)/(1.0299*(D/er0)**(5/3))) * scr)
                r0_act = []

                cutouts = [target.cutout(D, a, l, d, remove_piston=False) for a, l, d in zip(alts, atm, gen.metapupil_diameters)]
                for co in cutouts:
                    r0_act.append(phase_screen.calculate_r0(co, D, mask=mask))                    

                cutout = target.total_cutout(D, alts, atm, gen.metapupil_diameters)
                r0_act.append(phase_screen.calculate_r0(cutout, D, mask=mask))
                
                data[r0_total].append(phase_screen.calculate_r0(cutout, D))

                f.write(f"{r0[0]},{r0[1]},{r0[2]},{r0_total},{r0_act[0]},{r0_act[1]},{r0_act[2]},{r0_act[3]},")
                if row == row_start:
                    f.write(f"=SORT(H${row_start}:H${row_end})")
                f.write(f",\"=NORM.DIST(I{row},D{row},STDEV.S(H${row_start}:H${row_end}),TRUE)\"\n")


    df = pd.read_csv("data/tests/r0.csv")
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
    df.to_excel('data/tests/r0.xlsx', index=False)

    for k, v in data.items():
        # t_stat, p_value = stats.ttest_1samp(v, k)
        print(f"{k*100:.2f},{np.mean(v)*100:.2f},{np.mean(v)/k:.3f}")
    print("TEST")