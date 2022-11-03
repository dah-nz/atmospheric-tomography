from parameters import parameters
from simulation import Sim
import tomo_covariance
import tomo_projection
import tomo_projection_nowfs
import plot
from projection import projectionMatrix
from star import Star
import util

import concurrent.futures
import threading
from datetime import datetime

import numpy as np

data = []
lock = threading.Lock()

test_modes = []
test_guidestars = []
test_projMatrices = []
target_positions = []
test_asterisms = {}

file_name = "TEST_{}.data".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

def generate_test_targets():
    global target_positions

    # Azimuth settings (arcseconds)
    min_azi = 0
    max_azi = 300
    num_azi = 30

    # XY settings (radians)
    min_theta = 0
    max_theta = np.pi * 2
    num_theta = 18

    target_positions_azi = [util.arcsec2rad(x) for x in np.linspace(min_azi, max_azi, num_azi)]
    target_positions_theta = [x for x in np.linspace(min_theta, max_theta, num_theta)]
    for a in target_positions_azi:
        for t in target_positions_theta:
            target_positions.append((a, t))

def generate_test_target_modes(p):
    global test_modes

    num_tests = 30
    num_layers = len(p.layer_alt)
    num_modes = p.nModes
    num_ignores = p.modeIgnore

    # random modes
    for x in range(num_tests):
        layers = []
        for _ in range(num_layers):
            layers.append([0.0] * num_ignores + list(m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(num_modes) - 0.5)))
        test_modes.append([tuple(list([x for x in l])) for l in layers])

def generate_test_asterisms():
    global test_asterisms

    # Radii for testing (arcseconds)
    min_rad = 100
    max_rad = 300
    num_rad = 5

    radii = np.linspace(util.arcsec2rad(min_rad), util.arcsec2rad(max_rad), num_rad)

    # Number of stars, and rotation around zenith
    min_stars = 3
    max_stars = 5

    equal_spaced_stars = [np.linspace(0, 2*np.pi, x, endpoint=False) for x in range(min_stars, max_stars + 1)]
    random_single_radius_stars = [np.random.rand(x) * 2*np.pi for x in range(min_stars, max_stars + 1)]
    fully_random_stars = [list(zip(util.arcsec2rad((max_rad - min_rad) * np.random.rand(x) + min_rad), np.random.rand(x) * 2*np.pi)) for x in range(min_stars, max_stars + 1)]

    test_asterisms["equal"] = [[(a, t) for t in s] for s in equal_spaced_stars for a in radii]
    test_asterisms["random_single"] = [[(a, t) for t in s] for s in random_single_radius_stars for a in radii]
    test_asterisms["full_random"] = fully_random_stars

def main():
    np.random.seed(1)
    p = parameters()

    generate_test_targets()
    generate_test_target_modes(p)
    generate_test_asterisms()

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
                f.write("{},{},{}\n".format(index,ast_type,asterism))
                index += 1

        f.write("TEST_TARGET_MODES\n")
        for i, target_modes in enumerate(test_modes):
            f.write("{},{}\n".format(i,target_modes))

    print("Setting up test arguments")
    test_args = []
    for target_idx in range(len(target_positions)):
        for asterism_idx in range(len(test_guidestars)):
            for mode_idx in range(len(test_modes)):
                test_args.append((target_idx, asterism_idx, mode_idx))

    with open(file_name, 'a') as f:
        f.write("TEST_OUTPUTS\n")

    print("Running tests")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(sim_thread, test_args)
                
    print("FINISHED")


def sim_thread(args):
    target_idx, asterism_idx, mode_idx = args
    print("Test {}:{}:{} Running".format(target_idx, asterism_idx, mode_idx))
    global data
    global lock

    global test_projMatrices
    global test_modes
    global test_guidestars
    global target_positions

    projM = test_projMatrices[target_idx]
    target = target_positions[target_idx]

    gs = test_guidestars[asterism_idx]
    modes = test_modes[mode_idx]

    p = parameters()
    p.layer_modes = [np.matmul(np.linalg.pinv(pm), tm)[p.modeIgnore:] for pm, tm in zip(projM, modes)]
    p.targetPos = target
    sim = Sim(p, guidestars=gs)
    p.nGS = len(gs)

    t_nowfs = tomo_projection_nowfs.projection_nowfs(sim)
    lock.acquire()
    with open(file_name, 'a') as f:
        f.write("{},{},{},{}\n".format(target_idx, asterism_idx, mode_idx, list(t_nowfs)))
    data.append((target_idx, asterism_idx, mode_idx, t_nowfs, p.layer_modes))
    lock.release()
    print("Test {}:{}:{} Complete".format(target_idx, asterism_idx, mode_idx))

if __name__ == "__main__":
    main()