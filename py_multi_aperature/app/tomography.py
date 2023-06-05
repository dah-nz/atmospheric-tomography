from parameters import parameters
from simulation import Sim
import tomo_covariance
import tomo_projection
import tomo_projection_nowfs
import plot
from projection import projectionMatrix
from star import Star
import util
import phase_screen
import layer

import concurrent.futures
import threading
from datetime import datetime

import numpy as np

data = []
lock = threading.Lock()

file_name = "data/TEST_{}.data".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

NUM_TARGETS_AZIMUTH = 10
NUM_TARGETS_THETA = 12
NUM_PHASE_SCREENS = 30
NUM_ASTERISMS = 10

def generate_test_targets():
    target_positions = []
    azimuth_values = np.linspace(0, 300, NUM_TARGETS_AZIMUTH)
    theta_values = np.linspace(0, np.pi * 2, NUM_TARGETS_THETA, endpoint=False)

    for a in azimuth_values:
        a = util.arcsec2rad(a)
        for t in theta_values:
            target_positions.append((a, t))

    return target_positions

def generate_test_phase_screens(p):
    test_screens = []
    num_tests = NUM_PHASE_SCREENS
    num_layers = len(p.layer_alt)
    
    for x in range(num_tests):
        layers = []
        for layer in range(num_layers):
            seed = (x*num_layers + layer)
            layers.append((phase_screen.get_kolmogorov_screen(p.asize, p.D/p.r0, seed=seed), seed))
        test_screens.append([tuple(list([x for x in l])) for l in layers])
    
    return test_screens

def generate_test_asterisms():
    test_asterisms = {}

    # Radii for testing (arcseconds)
    min_rad = 50
    max_rad = 500
    num_rad = NUM_ASTERISMS

    radii = np.linspace(util.arcsec2rad(min_rad), util.arcsec2rad(max_rad), num_rad)

    # Number of stars
    min_stars = 3
    max_stars = 9

    equal_spaced_stars = [np.linspace(0, 2*np.pi, x, endpoint=False) for x in range(min_stars, max_stars + 1)]
    fully_random_stars = [list(zip(util.arcsec2rad((max_rad - min_rad) * np.random.rand(x) + min_rad), np.random.rand(x) * 2*np.pi)) for x in range(min_stars, max_stars + 1)]

    test_asterisms["equal"] = [[(a, t) for t in s] for s in equal_spaced_stars for a in radii]
    test_asterisms["random"] = fully_random_stars

    return test_asterisms

def main():
    np.random.seed(1)
    p = parameters()

    target_positions = generate_test_targets()
    test_screens = generate_test_phase_screens(p)
    test_asterisms = generate_test_asterisms()
    test_guidestars = []
    test_projMatrices = []

    base_sim = Sim(p)
    def proj(ts, layer):
        x, y = ts.position(layer.altitude)
        s = base_sim.p.asize / layer.dim
        P = projectionMatrix((x/base_sim.p.D, y/base_sim.p.D), s, base_sim.p.nModes + base_sim.p.modeIgnore)
        return P.T

    with open(file_name, 'w') as f:
        # Calculate target position and projection matrices
        print("Generating Projection Matrices")
        f.write("TEST_TARGET_POSITIONS\n")
        for i, target in enumerate(target_positions):
            test_projMatrices.append([(proj(Star(target[0], target[1]), layer)) for layer in base_sim.layers])
            f.write("{},{}\n".format(i, target))

        # Calculate asterism positions
        print("Generating Asterisms")
        f.write("TEST_ASTERISMS\n")
        index = 0
        for ast_type, ast in test_asterisms.items():
            for _, asterism in enumerate(ast):
                test_guidestars.append([Star(t, p) for t, p in asterism])
                f.write("{},\"{}\",{}\n".format(index,ast_type,asterism))
                index += 1

        f.write("TEST_TARGET_SCREEN_SEED\n")
        for i, screens in enumerate(test_screens):
            f.write("{},{}\n".format(i,[m[1] for m in screens]))

    test_data = {"pm": test_projMatrices, "pos": target_positions, "gs": test_guidestars, "ps": test_screens}

    def check_valid_asterism(target_idx, asterism_idx):
        target = test_data['pos'][target_idx]
        target = Star(target[0], target[1])
        asterism = [x.position(1000) for x in test_data['gs'][asterism_idx]]
        return util.bounded(asterism, target.position(1000))
        
    print("Setting up test arguments")
    test_args = []
    for target_idx in range(len(target_positions)):
        for asterism_idx in range(len(test_guidestars)):
            if check_valid_asterism(target_idx, asterism_idx):
                for screen_idx in range(len(test_screens)):
                    test_args.append((target_idx, asterism_idx, screen_idx, test_data))

    with open(file_name, 'a') as f:
        f.write("TEST_OUTPUTS\n")

    print("Running tests")
    # for args in test_args:
    #     sim_thread(args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(sim_thread, test_args)
                
    print("FINISHED")


def sim_thread(args):
    target_idx, asterism_idx, screen_idx, test_data = args
    test_str = "Test {}:{}:{}".format(target_idx, asterism_idx, screen_idx)
    print(test_str + " Running")
    global data
    global lock

    projM = test_data["pm"][target_idx]
    target = test_data['pos'][target_idx]

    gs = test_data['gs'][asterism_idx]
    phaseScreen = test_data['ps'][screen_idx]

    p = parameters()
    p.targetPos = target
    ps = [p[0] for p in phaseScreen]
    sim = Sim(p, guidestars=gs, phaseScreens=ps)
    p.nGS = len(gs)

    target_star = Star(target[0], target[1])
    layers = [layer.Layer(p, gs, target_star, l, phaseScreen=screen) for l, screen in zip(p.layer_alt, ps)]

    target_cut_outs = [l.cut_out(p, target_star) for l in layers]
    # print(test_str + " Cut Outs Defined")

    if all([np.shape(s)[0] > 0 for s in target_cut_outs]):
        target_modes = [phase_screen.getModes(cut_out, p.nModes) for cut_out in target_cut_outs]
        target_modes = np.sum(target_modes, 0)
        # print(test_str + " Target Modes Calculated")
        layer_modes = [phase_screen.getModes(l.phaseScreen, p.nModes) for l in sim.layers]
        sim.p.layer_modes = layer_modes

        # print("Size: {}".format(np.shape(target_modes)))

        t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)

        lock.acquire()
        with open(file_name, 'a') as f:
            f.write("{},{},{},{},{}\n".format(target_idx, asterism_idx, screen_idx, list(t_nowfs), list(target_modes)))
        data.append((target_idx, asterism_idx, screen_idx, t_nowfs, p.layer_modes))
        lock.release()

        print(test_str + " Complete")
    else:
        print(test_str + " Incomplete - Invalid cut out size")

if __name__ == "__main__":
    main()