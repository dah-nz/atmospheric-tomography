class Tomography():
    """
    Interface class that implements a tomography algorithm
    This class should not have direct knowledge of phase screen information at all

    Methods
    -------
    run:
        runs tomographic algorithm using beacon coefficients, and positions of guide beacons and target
    """
    def __init__(self, basis, diameter, n_modes, aperture_size=256):
        """
        Parameters
        ----------
        layers:
            altitudes of each layer
        basis:
            class with interface to a set of basis functions 
        modes:
            number of modes to do tomography over
        diameter:
            telescope diameter
        r0:
            fried parameter for atmosphere
        aperture_size:
            size of telescope aperture in pixels 
        """
        self.basis = basis
        self.n_modes = n_modes
        self.D = diameter
        self.aperture_size = aperture_size

    def run(self, beacons, target, beacon_coeffs):
        """
        Run tomographic algorithm
        """
        raise NotImplementedError()
    
    def get_layer_shapes(self, beacons):
        mD = self._metapupil_diameter(beacons)
        metapupil_diameters = [mD[a] for a in self.altitudes]
        layer_pixels = [int(d/self.D * self.aperture_size + 0.5) for d in metapupil_diameters]
        layer_grids = [(px, px) for px in layer_pixels]
        self.metapupil_diameters = metapupil_diameters
        self.layer_grids = layer_grids
        return metapupil_diameters, layer_grids
