import numpy as np
import phase_screen
import util

class Layer:
    def __init__(self, p, guideStars, target, altitude, zModes=None):
        self.update(p, guideStars, target, altitude, zModes)
    def update(self, p, guideStars, target, altitude, zModes=None):
        self.guideStars = guideStars
        self.target = target
        self.altitude = altitude
        
        # calculate radius of metapupil 
        self.r_outer = 0                            # outer radius of metapupil
        r0 = altitude * np.tan(target.theta)        # distance of target centre from azimuth
        if r0 > self.r_outer:
            self.r_outer = r0
        for gs in self.guideStars:
            rA = altitude * np.tan(gs.theta)        # distance of guide star centre from azimuth
            if rA > self.r_outer:
                self.r_outer = rA                        
        self.r_outer += (p.D / 2)              # Calculate extent of footprint projected onto layer (diameter is of telescope aperture)

        # calculate dimension of layer with the same pixel scale as the telescope aperture
        self.dim = int((p.asize / p.D) * self.r_outer + 0.5) * 2

        # Generate phase screen for the layer, based on zernike modes
        if zModes is not None:
            self.modes = zModes
            self.phaseScreen = phase_screen.createFromModes(self.modes, self.dim, p.modeIgnore)
        
        # _, Minv = generateMMatrices(rT * asize) # Geometric WFS sensor matrices - defined based on telescope aperture size
        # convert radial position to cartesian, normalised to metapupil size
        # for rA, phi in gs_rad:
            # rA /= r_outer
            # pos = pol2cart(rA * (asize / 2), phi)
            # pMatrix = projectionMatrix(pos, rT/2, nModes, modeOffset)  # Generate projection matrix for each guide star
            # wF = GeoWFS.circle(rT * (asize / 2), asize, pos)
            # guideStar = StarAberration(pos, rT, wF, pMatrix)
            # gs.append(guideStar)

        # gs.append(gs.pop(0))

        # generate projection matrix for target
        # tMatrix = projectionMatrix((0, 0), rT, nModes, modeOffset)
        # tWF = util.circle(rT * (asize / 2), asize)
        # r0 /= r_outer
        # pos = pol2cart(r0 * (asize / 2), tg_azi_pos[1])
        # target = StarAberration(pos, rT, tWF, tMatrix)

        # return Layer(phase, Minv, altitude, gs, target, modes, rT)

def generate_all(p, gs, tgt):
    layers = []
    actual_modes = np.zeros((p.nModes))
    for alt, modes in zip(p.layer_alt, p.layer_modes):
        layer = Layer(p, gs, tgt, alt, modes)
        layers.append(layer)
        # layer.number = layer_number
        # run_layer(layer)
        # P = projectionMatrix(tPos, layer.rT, nModes, modeOffset)
        # actual_modes += np.matmul(P, modes)
    return layers

def main():
    import matplotlib.pyplot as plt
    import matplotlib
    from star import Star
    import phase_screen
    # matplotlib.use("pgf")
    stars = [Star(util.arcsec2rad(200), n) for n in [0., 2 * np.pi / 3, 4 * np.pi / 3]]
    target = Star(0, 0)
    class p_c:
        def __init__(self):
            self.D = 1
            self.asize = 256
    p = p_c()
    l1 = Layer(p, stars, target, 1000)
    l2 = Layer(p, stars, target, 8000)
    l1_ps = np.float64(phase_screen.kolmogorov(p.asize, 15))
    l2_ps = np.float64(phase_screen.kolmogorov(p.asize, 15))

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(5.5, 4)
    for ax in axs:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    axs[0].set_title("Layer at 1000m")
    axs[1].set_title("Layer at 8000m")
    # axs[0].imshow(l1_ps, cmap='GnBu')
    # axs[1].imshow(l2_ps, cmap='GnBu')

    # Plot Target Location
    for ax, layer in zip(axs, [l1, l2]):
        x, y = target.position(layer.altitude)
        tx = (x * p.asize / p.D) * (p.asize / layer.dim) + (p.asize / 2)
        ty = (y * p.asize / p.D) * (p.asize / layer.dim) + (p.asize / 2)
        util.plotcirc(ax, (tx, ty), ((p.asize ** 2) / layer.dim) / 2, 'r:')
        # Plot Guide stars
        for gs in layer.guideStars:
            x, y = gs.position(layer.altitude)
            gx = (x * p.asize / p.D) * (p.asize / layer.dim) + (p.asize / 2)
            gy = (y * p.asize / p.D) * (p.asize / layer.dim) + (p.asize / 2)
            util.plotcirc(ax, (gx, gy), ((p.asize ** 2) / layer.dim) / 2, 'k--')

    # util.plotcirc(axs[0], util.cart2imagespace(l1.dim, target.position(1000)), p.asize / 2, 'r--', alpha=0.7)
    # util.plotcirc(axs[1], util.cart2imagespace(l2.dim, target.position(8000)), p.asize / 2, 'r--', alpha=0.7)
    # for star in stars:
    #     util.plotcirc(axs[0], util.cart2imagespace(l1.dim, star.position(1000)), p.asize / 2, 'k--', alpha=0.7)
    #     util.plotcirc(axs[1], util.cart2imagespace(l2.dim, star.position(8000)), p.asize / 2, 'k--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig("img/layer_asterism.pgf")
    plt.show()

if __name__ == "__main__":
    main()

