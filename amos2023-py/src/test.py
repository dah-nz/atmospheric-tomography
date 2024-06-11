import glob
import os

start_time = "20240611_155529"

results = f"data/models/results/*.dat"
files = glob.glob(results)
for file in files:
    f = file.split('\\')
    nm = f[1].split(".")
    f2 = f"{f[0]}/old/{nm[0]}_{start_time}.{nm[1]}"
    os.rename(file, f2)

quit()

import numpy as np
import matplotlib.pyplot as plt

from tomography import tomo_LSQ
from basis import zernike, DCT
from tomography import beacon, phase_screen
from util import util

D = 1.0
ap_sz = 64
modes = 25
ast_gamma = 15
alts = [1000, 3000, 10000]

r0 = 0.07 # m
rng = np.random.default_rng()

# basis_fn = zernike.Zernike(modes)
basis_fn = DCT.DCT(modes)

cmap = "BrBG"

# tomo_LSQ.PADDING = 0



# beacon_list = {ast_gamma: [beacon.Beacon(t, ast_gamma) for t in np.linspace(0, 2 * np.pi, 3, endpoint=False)] for ast_gamma in np.linspace(1, 15, 2)}
beacon_list = {}
beacon_list["random"] = [beacon.Beacon(t, g, mask=basis_fn.mask) for t, g in zip(rng.uniform(0, 2*np.pi, size=3), rng.gamma(ast_gamma, 1, size=3))]

target = beacon.Beacon(0, 0, mask=basis_fn.mask)

for g, beacons in beacon_list.items():
    md = util.get_metapupil_diameters(alts, beacons, D)
    print(md)

    szs = [int(ap_sz * (d / D) + 1) for d in md]
    for a, d, sz in zip(alts, md, szs):
        print(f"D: {D}, a: {a:5}, d: {d:7.4f}, sz: {sz:4}")

    mask = lambda x: util.circle(x / 2, x) if basis_fn.mask else np.ones((x, x))
    gens = [phase_screen.PhaseScreenGenerator(s, D, r0, rng) for s in szs]
    cutout = (ap_sz, ap_sz)
    screens = [mask(s) * g.generate(1, cutout, "all")[0] for g, s in zip(gens, szs)]

    # screen_modes = np.zeros(modes)
    # screen_modes[1] = 4
    # screens = [util.circle(s / 2, s) * basis_fn.compose((s, s), screen_modes) for s in szs]
    # screens = [mask(s) * (sc - np.mean(sc)) for sc, s in zip(screens, szs)]

    # targets = []
    # guidestars = []

    target_cutout = []
    gs_cutouts = []

    for l, screen in enumerate(screens):
        fig, axs = plt.subplots(3, 5)
        fig.suptitle(f"Layer {l + 1}")

        gs_cutouts.append([])

        vmin = np.amin(screen)
        vmax = np.amax(screen)
        vmax = max(vmax, abs(vmin))
        vmin = -vmax

        axs[0, 0].imshow(screen, vmin=vmin, vmax=vmax, cmap=cmap)
        recons = mask(screen.shape[0]) * basis_fn.compose(screen.shape, basis_fn.decompose(screen))
        axs[1, 0].imshow(screen, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2, 0].imshow(screen - recons, vmin=vmin, vmax=vmax, cmap=cmap)

        cs = "krbg"
        
        for i, b in enumerate([target] + beacons):
            sz = szs[l]
            pos = util.cart2imagespace(sz, md[l], np.array(b.position(alts[l])))
            rad = ap_sz / 2
            print(f"pos: {pos}, rad: {rad}, sz: {sz}")
            title = f"GS{i}" if i > 0  else "Target"
            util.plotcirc(axs[1, 0], pos, rad, cs[i], label=title)

            axs[1, 0].plot([sz/2, pos[0]], [sz/2, pos[1]], cs[i])

            cutout = b.cut_out(D, alts[l], screen, md[l])
            if i == 0:
                target_cutout.append(cutout)
            else:
                gs_cutouts[l].append(cutout)

            axs[0, i+1].set_title(title)
            axs[0, i+1].imshow(cutout, vmin=vmin, vmax=vmax, cmap=cmap)
            
            coeffs = basis_fn.decompose(cutout)

            dc = mask(cutout.shape[0]) * basis_fn.compose(cutout.shape, coeffs)

            axs[1, i+1].imshow(dc, vmin=vmin, vmax=vmax, cmap=cmap)
            axs[2, i+1].imshow(cutout - dc, vmin=vmin, vmax=vmax, cmap=cmap)
        
        axs[1, 0].plot([0, sz],[sz/2, sz/2], 'y:')
        axs[1, 0].plot([sz/2, sz/2],[0, sz], "y:")
        axs[1, 0].plot([0, sz], [sz/2, sz], "k:")

        axs[1, 0].legend()
        plt.tight_layout()

    # sum target coeffs along layers
    target_cutout = np.array(target_cutout)
    target_ps = np.sum(target_cutout, axis=0)
    target_ps = target_ps - np.average(target_ps)
    target_coeffs = basis_fn.decompose(target_ps)
    
    # sum guide star coeffs along layers and flatten
    gs_cutouts = np.array(gs_cutouts)
    print(gs_cutouts.shape)
    gs_ps = np.sum(gs_cutouts, axis=0)
    print(gs_ps.shape)
    gs_ps = [ps - np.average(ps) for ps in gs_ps]
    guide_coeffs = [basis_fn.decompose(ps) for ps in gs_ps]
    
    guide_coeffs = np.ravel(guide_coeffs)

    def proj(b, l):
        a = alts[l]
        sz = (szs[l], szs[l])
        L = []

        for m in range(modes):
            coeffs = np.zeros(modes)
            coeffs[m] = 1

            scr = basis_fn.compose(sz, coeffs)
            co = b.cut_out(D, a, scr, md[l])
            dco = basis_fn.decompose(co)
            L.append(dco)
        eps = 1e-12
        P = np.array(L)
        P = (np.abs(P) > eps) * P
        return P

    fig, axs = plt.subplots(len(alts), len(beacons) + 1)
    fig.suptitle("Projection matrices")
    for l, a in enumerate(alts):
        for i, b in enumerate([target] + beacons):
            lbl = f"GS {i}" if i > 0 else "Target"
            if len(alts) == 1:
                axs[i].imshow(proj(b, l), cmap=cmap)
                axs[i].set_title(f"Layer {l+1}; {lbl}")
            else:
                axs[l, i].imshow(proj(b, l), cmap=cmap)
                axs[l, i].set_title(f"Layer {l+1}; {lbl}")


    # build A matrix
    A = np.concatenate([np.concatenate([proj(beacon, l) for beacon in beacons], 0) for l, _ in enumerate(alts)], 1)
    print(f"cond(A) = {np.linalg.cond(A)}")

    # build T matrix
    T = np.concatenate([proj(target, l) for l, _ in enumerate(alts)], 1)
    print(f"cond(T) = {np.linalg.cond(T)}")

    M = np.linalg.pinv(np.matmul(A, np.linalg.pinv(T)))
    # M = np.matmul(T, np.linalg.pinv(A))
    M0 = np.matmul(T, np.linalg.pinv(A))

    print(f"cond(M) = {np.linalg.cond(M)}, cond(M0) = {np.linalg.cond(M0)}")
    Wt = np.matmul(M, guide_coeffs)

    print(Wt, target_coeffs)

    tgt_scr = np.sum(target_cutout, axis=0)
    vmin = np.amin(tgt_scr)
    vmax = np.amax(tgt_scr)

    fig, axs = plt.subplots(2, 3)
    fig.suptitle("Tomography Results")
    axs[0, 0].imshow(tgt_scr, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0, 0].set_title("Target Wavefront")
    act = basis_fn.compose((ap_sz, ap_sz), target_coeffs)
    axs[0, 1].imshow(act, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0, 1].set_title("Best Estimate")
    axs[1, 1].imshow(tgt_scr - act, vmin=vmin, vmax=vmax, cmap=cmap)
    pre = basis_fn.compose((ap_sz, ap_sz), Wt)
    axs[0, 2].imshow(pre, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0, 2].set_title("Predicted")
    axs[1, 2].imshow(tgt_scr - pre, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1, 0].plot(Wt, label="Tomo Pred")
    axs[1, 0].plot(target_coeffs, label="Actual")
    axs[1, 0].legend()


for g, beacons in beacon_list.items():
    md = util.get_metapupil_diameters(alts, beacons, D)
    
    szs = [int(ap_sz * (d / D) + 1) for d in md]
    
    gens = [phase_screen.PhaseScreenGenerator(s, D, r0, rng) for s in szs]
    cutout = (ap_sz, ap_sz)
    screens = [g.generate(1, cutout, "all")[0] for g in gens]

    lsq = tomo_LSQ.LSQ(basis_fn, D, modes, ap_sz)

    coeffs = util.get_coeffs_for_asterism(beacons, basis_fn, screens, alts, D, md)
    print(coeffs.shape)

    tgt = beacon.Beacon(0, 0, mask=basis_fn.mask)
    tgt_coeffs = util.get_coeffs_for_cutout(tgt, basis_fn, screens, alts, D, md)
    print(tgt_coeffs.shape)

    pred_coeffs = lsq.run(beacons, tgt, coeffs, alts)

    print(pred_coeffs.shape)

    plt.figure()
    plt.plot(tgt_coeffs, label="Target Coeffs")
    plt.plot(pred_coeffs, label="Predicted Coeffs")
    plt.title("Tomography Coefficient Performance Comparison")
    plt.legend()

    fig, axs = plt.subplots(2, 3)
    fig.suptitle(f"Asterism angle = {g}")

    actual_screen = util.get_cutout(tgt, screens, alts, D, md)
    vmin = np.amin(actual_screen)
    vmax = np.amax(actual_screen)
    axs[0, 0].imshow(actual_screen, vmin=vmin, vmax=vmax, cmap=cmap)

    tgt_scr = basis_fn.compose((ap_sz, ap_sz), tgt_coeffs)
    pred_scr = basis_fn.compose((ap_sz, ap_sz), pred_coeffs)

    axs[0, 1].imshow(tgt_scr, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0, 2].imshow(pred_scr, vmin=vmin, vmax=vmax, cmap=cmap)

    axs[1, 0].imshow(actual_screen - tgt_scr, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1, 1].imshow(actual_screen - pred_scr, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1, 2].imshow(tgt_scr - pred_scr, vmin=vmin, vmax=vmax, cmap=cmap)

# tomo_LSQ.main()

plt.show()


### Asterism generation code test below

"""
import numpy as np
import matplotlib.pyplot as plt
import util

rng = np.random.default_rng()

offset = 5 * (np.pi / 180)

t1 = rng.uniform(low=0.0, high=2*np.pi)
t2 = rng.uniform(low=t1 + offset, high=2*np.pi + t1 - offset)
if t2 > t1 + np.pi:
    t1 += 2* np.pi
    t1, t2 = t2, t1
t3 = rng.uniform(low=min(t1, t2)-offset, high=max(t1, t2)+offset)

plt.plot(np.cos([t1, t1 + np.pi]), np.sin([t1, t1 + np.pi]), 'b--', label='1')
plt.plot(np.cos([t2, t2 + np.pi]), np.sin([t2, t2 + np.pi]), 'r--', label='2')

plt.plot(np.cos([t1, t2, t3+np.pi, t1]), np.sin([t1, t2, t3+np.pi, t1]), 'x-')

print(t1, t2, t3)
# print(t1i, t2i)

plt.legend()
plt.figure()

# plt.show()
# quit()

star_gamma = np.zeros(shape=(5,3))

st0 = rng.uniform(low=0, high=2*np.pi, size=(star_gamma.shape[0], 1))
st1 = rng.uniform(low=st0 + offset, high=2*np.pi + st0 - offset, size=(star_gamma.shape[0], 1))
print(st0, st1)
st0[st1 > (st0 + np.pi)] = st0[st1 > (st0 + np.pi)] + 2 * np.pi

st2 = rng.uniform(low=np.min([st0, st1], axis=0)-offset, high=np.max([st0, st1], axis=0)+offset) + np.pi
st = np.concatenate([st0, st1, st2], axis=1)

print(st)

plt.plot([0], [0], 'ko')

for i in range(st.shape[0]):
    s = st[i, :]
    s = np.concatenate((s, [s[0]]))
    x = np.cos(s)
    y = np.sin(s)
    plt.plot(x, y, 'x-', label=f"{i+1}")

plt.legend()

plt.show()
"""