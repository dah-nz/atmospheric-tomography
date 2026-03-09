import numpy as np
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Asterism import Asterism
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Zernike import Zernike
from OOPAO.ShackHartmann import ShackHartmann

import tomoAO
from tomoAO.Reconstruction.reconClassType import tomoReconstructor

class OOPAO_Tomo():
    def __init__(self, asterism, r0, n_modes, altitudes, D=1.8, fov=120, res=256, n_subaps=16, seed=1, display=False):
        self._beacons = asterism
        self.r0 = r0

        self.tel = Telescope(diameter=D, resolution=res, fov=fov)
        self.ast = Asterism([Source(optBand='V', magnitude=b.magnitude, coordinates=[b._gamma_arcsec, np.degrees(b.theta)]) for b in self._beacons])

        assert self.tel.fov >= 2 * max([b._gamma_arcsec for b in self._beacons])

        self.ast * self.tel

        self._n_subaps = n_subaps
        n_pix_per_subap = int(res / self._n_subaps)


        N = len(altitudes)

        self.atm = Atmosphere(telescope=self.tel,
                              r0=r0,
                              L0=25,
                              fractionalR0=[1./N] * N,
                              windSpeed=[1.]*N,
                              windDirection=[0]*N,
                              altitude=altitudes)
        
        self.target = Source(optBand='V',
                             magnitude=0.0,
                             altitude=np.inf,
                             coordinates=[0, 0])
        
        self.science = Source(optBand='V',
                              magnitude=10.0,
                              coordinates=[0, 0])
        
        self.atm.initializeAtmosphere(telescope=self.tel, compute_covariance=False)
        self.atm.generateNewPhaseScreen(seed=seed)
        self.atm.relay(self.target)

        if display == True:
            self.atm.display_atm_layers()

        self.ast * self.tel

        # self.Z = Zernike(self.tel, J=n_modes)
        # self.Z.computeZernike(self.tel)
        # modes = np.reshape(self.Z.modesFullRes, (res * res, n_modes))

        self.wfs = ShackHartmann(nSubap=self._n_subaps, 
                                 telescope=self.tel, 
                                 lightRatio=0.5, 
                                 is_geometric=False,
                                 shannon_sampling=False,
                                 n_pixel_per_subaperture=n_pix_per_subap)
        
        self.ast ** self.tel * self.wfs

        z = Zernike(self.tel, N)
        zernike_modes = z.computeZernike(self.tel)
        self.dm = DeformableMirror(telescope=self.tel,
                                   nSubap=self._n_subaps,
                                   modes=zernike_modes)

        M = np.zeros(self.dm.nAct**2,dtype=bool)
        M[self.dm.validAct]=True
        unfiltered_act_mask = M.reshape(self.dm.nAct,self.dm.nAct)
        self.dm.unfiltered_act_mask = unfiltered_act_mask



        self.target * self.tel
        self.science * self.tel

        os = 2
        self.config_vars = {
            'os': os,
            'nSubaperture': self._n_subaps,
            'mechanicalCoupling': self.dm.mechCoupling,
            'dm_resolution': os * self._n_subaps + 1,
            'resolution': self.tel.resolution
        }

    def run(self):
        aoSys = tomoAO.Simulation.AOSystem(self.config_vars,
                                   tel=self.tel,        # telescope
                                   ngs=self.target,     # natural guide star (giving optimization direction)
                                   lgsAst=self.ast,     # asterism
                                   sciSrc=self.science,  # science source
                                   atm=self.atm,        # atmosphere
                                   dm=self.dm,          # deformable mirror
                                   wfs=self.wfs)        # WFS
        
        rec = tomoReconstructor(aoSys=aoSys,
                                alpha=10,
                                os=self.config_vars['os'],
                                indexation='xxyy',
                                remove_TT_F=False,
                                filter_subapertures=False)
        
        self.reconstructor = rec.reconstructor.copy()
        self.atm.r0 = self.r0

        self.ast ** self.tel * self.wfs

        self.wfs.set_slopes_units(src=self.ast, tomographic_reconstructor=self.dm.modes@self.reconstructor)

        self.ast ** self.atm * self.tel * self.wfs

        self.wfs_signal = np.hstack(self.wfs.signal)

        self.dm.coefs = -self.reconstructor@self.wfs_signal

        self.science ** self.atm * self.tel
        original = self.science.phase.copy()
        original = original # - np.mean(original, where=np.array(self.tel.pupil, dtype=bool))

        self.science ** self.atm * self.tel * self.dm
        reconstruction = self.science.phase.copy()
        reconstruction = reconstruction # - np.mean(reconstruction, where=np.array(self.tel.pupil, dtype=bool))

        # self.science.OPD = self.science.OPD - np.mean(self.science.OPD)

        return original, reconstruction