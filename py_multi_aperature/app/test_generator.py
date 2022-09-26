import numpy as np
import util

def target_modes(p, n):
    test_modes = []

    num_layers = len(p.layer_alt)
    num_modes = p.nModes
    num_ignores = p.modeIgnore

    # random modes
    for _ in range(n):
        layers = []
        for _ in range(num_layers):
            layers.append([0.0] * num_ignores + list(m * np.exp(-(i+1)/10) for i, m in enumerate(np.random.rand(num_modes) - 0.5)))
        test_modes.append([tuple(list([x for x in l])) for l in layers])

    return test_modes

def target_projectionMatrices(sim, t):
    from projection import projectionMatrix
    from star import Star
    def proj(sim, ts, layer):
        x, y = ts.position(layer.altitude)
        s = sim.p.asize / layer.dim
        P = projectionMatrix((x/sim.p.D, y/sim.p.D), s, sim.p.nModes + sim.p.modeIgnore)
        return P.T
    
    T = []
    for p in t:
        T.append([proj(sim, Star(p[0], p[1]), l) for l in sim.layers])

    return T

def radii(min_rad, max_rad, n, arcseconds=True):
    if arcseconds:
        return np.linspace(util.arcsec2rad(min_rad), util.arcsec2rad(max_rad), n)
    else: # radians
        return np.linspace(min_rad, max_rad, n)

def asterisms(stars, radii):
    from star import Star
    test_asterisms = {}

    equal_spaced_stars = [np.linspace(0, 2*np.pi, x, endpoint=False) for x in stars]
    random_single_radius_stars = [np.random.rand(x) * 2*np.pi for x in stars]
    fully_random_stars = [list(zip(((np.amax(radii) - np.amin(radii)) * np.random.rand(x) + np.amin(radii)), np.random.rand(x) * 2*np.pi)) for x in stars]
    
    test_asterisms["equal"] = [[(a, t) for t in s] for s in equal_spaced_stars for a in radii]
    test_asterisms["random_single"] = [[(a, t) for t in s] for s in random_single_radius_stars for a in radii]
    test_asterisms["full_random"] = fully_random_stars

    test_guidestars = {}
    test_guidestars["equal"] = []
    test_guidestars["random_single"] = []
    test_guidestars["full_random"] = []
    for ast_type, ast in test_asterisms.items():
            for _, asterism in enumerate(ast):
                test_guidestars[ast_type].append([Star(t, p) for t, p in asterism])

    return test_guidestars