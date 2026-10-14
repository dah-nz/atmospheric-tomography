import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Asterism import Asterism
from OOPAO.tools.tools import crop
import tomoAO
from OOPAO.tools.displayTools import cl_plot, displayMap

from OOPAO.Zernike import Zernike


#%% ### Telescope ###

n_subaperture = 10
n_pixel_per_subaperture = 8
 
tel = Telescope(diameter          = 1.8,   # diameter [m]
               resolution         = n_pixel_per_subaperture * n_subaperture,  # resolution [pix.]
               centralObstruction = 0.0,   # central obstruction
               samplingTime       = 0.001, # sampling time [s]
               fov                = 30)    # FOV of telescope [arcsec]

plt.figure()
plt.imshow(tel.pupil)
plt.title("Telescope pupil",fontsize=16,pad=10)
plt.colorbar()
# plt.show()

#%% ### Asterism ###

from OOPAO.Source import Source

from OOPAO.Asterism import Asterism

n_ngs = 3
ngs_zenith = [15] * n_ngs
ngs_azimuth = np.linspace(0,360,n_ngs,endpoint=False)

ngs_asterism = Asterism([Source(optBand='V',magnitude=0, coordinates=[ngs_zenith[kLgs], ngs_azimuth[kLgs]]) for kLgs in range(n_ngs)])
    
# sensing wavelength of the WFS
sensing_wavelength  = ngs_asterism.src[0].wavelength

# telescope fov needs to be in accordance with the asterism used
assert tel.fov >= 2*ngs_zenith[0]

ngs_asterism*tel



#%%

r0 = 0.06

atm = Atmosphere(telescope=tel,  # Telescope
                  r0=r0,  # Fried Parameter [m]
                  L0=25,  # Outer Scale [m]
                  fractionalR0=[0.8824,0.0294,0.0882],  # Cn2 Profile
                  windSpeed=[0,0,0],  # Wind Speed in [m]
                  windDirection=[0,0,0],  # Wind Direction in [degrees]
                  altitude=[1000,5000,10000]) # Altitude Layers in [m]

# atm = Atmosphere(telescope=tel,  # Telescope
#                   r0=r0,  # Fried Parameter [m]
#                   L0=25,  # Outer Scale [m]
#                   fractionalR0=[1],  # Cn2 Profile
#                   windSpeed=[10],  # Wind Speed in [m]
#                   windDirection=[0],  # Wind Direction in [degrees]
#                   altitude=[0]) # Altitude Layers in [m]
atm.initializeAtmosphere(telescope=tel)

atm.display_atm_layers()

#%%

ngs_asterism**tel

# diffractive SH WFS (is_geometric = False), with Nyquist sampled spots
wfs = ShackHartmann(telescope          = tel,
                      nSubap             = n_subaperture,
                      lightRatio         = 0.5,
                      is_geometric       = False,
                      shannon_sampling   = False,
                      threshold_cog      = 0.01,
                      n_pixel_per_subaperture=n_pixel_per_subaperture)

#%%
ngs_asterism**tel*wfs
displayMap(wfs.cam.frame,axis=0)
assert wfs.is_geometric == False

displayMap(wfs.reference_signal_2D,axis=0)
#%%

ngs_asterism**tel

# geometric SH WFS (is_geometric = True)
wfs_geom = ShackHartmann(telescope     = tel,
                      nSubap             = n_subaperture,
                      lightRatio         = 0.5,
                      is_geometric       = True,
                      shannon_sampling   = True,
                      threshold_cog      = 0.01)

assert wfs_geom.is_geometric == True

#%%

### Deformable mirror ###

z = Zernike(tel, 32)
zernike_modes=z.computeZernike(tel)

dm = DeformableMirror(telescope = tel,
                      nSubap = n_subaperture,
                      modes=zernike_modes)

print("DM pitch: {} m".format(dm.pitch))
print(tel.D/(dm.nAct-1))

## DM unfiltered actuator mask --> needed for the tomographic reconstruction ##
M = np.zeros(dm.nAct**2,dtype=bool)
M[dm.validAct]=True
unfiltered_act_mask = M.reshape(dm.nAct,dm.nAct)

plt.imshow(unfiltered_act_mask)
plt.title("DM unfiltered actuator mask",fontsize=14,pad=10)
plt.colorbar()
# plt.show()

dm.unfiltered_act_mask = unfiltered_act_mask

#%%

# natural guide star giving the optimization direction we want
# Giving coordinates [0,0] to ngs, means we are optimizing the tomographic reconstruction in this direction
target = Source(optBand='V',        # Optical band (see photometry.py)
             magnitude=14.0,       # Source Magnitude
             altitude=np.inf,     # altitude
             coordinates=[0, 0])  # Source coordinated [arcsec,deg]

# Define a scientific source:
science = Source(optBand='H', magnitude=0)

target*tel
science*tel

#%% ### Tomography ###
# Dictionary to be filled with parameters necessary for the tomographic reconstructor
config_vars = {}

# oversampling factor
config_vars["os"] = 2
# number of subapertures for the SH wfs
config_vars["nSubaperture"] = n_subaperture
# mechanical coupling of the Deformable mirror
config_vars['mechanicalCoupling'] = dm.mechCoupling
# resolution for the reconstructed phase
config_vars["dm_resolution"] = config_vars["os"] * config_vars["nSubaperture"] + 1
# resolution of the original simulated phase screens
config_vars["resolution"] = tel.resolution

# AO system. We give the different AO objects created in OOPAO as input
aoSys = tomoAO.Simulation.AOSystem(config_vars,
                                   tel=tel,         # telescope
                                   ngs=target,         # natural guide star (giving optimization direction)
                                   lgsAst=ngs_asterism,  # asterism
                                   sciSrc=science,   # science source
                                   atm=atm,         # atmosphere
                                   dm=dm,
                                   wfs=wfs)       # SH WFS

#%% ## Spatio-angular Tomographic reconstructor ###

from tomoAO.Reconstruction.reconClassType import tomoReconstructor

inital_time = time.time()

rec = tomoReconstructor(aoSys=aoSys,                  # AO system object (as crea:ted before)
                        alpha=10,                     # constant used to compute the noise covariance
                        os=config_vars["os"],         # oversampling factor (used to compute the reconstruction grid)
                        indexation="xxyy",            # related with slopes ordering (in oopao they are xxyy)
                        remove_TT_F = False,          # wether or not to remove TT from reconstruction
                        filter_subapertures = False)  # wether or not to filter SH subapertures (in simulation the
                                                      # valid subapertures are constant, so this is set False
                                                      
final_time = time.time()

print("Time spent to build tomographic reconstructor: {:.2f} seconds".format(final_time-inital_time))

reconstructor = rec.reconstructor.copy()
atm.r0 = r0 # this is important since the atm.r0 is changed inside the rec function
plt.figure()
plt.imshow(reconstructor)
plt.title("Spatio-angular reconstructor",fontsize=14,pad=10)
plt.colorbar()
# plt.show()
#%% Calibration of the SH WFS units to adapt to LGS gains, sampling, etc WITH tomographic reconstructor

ngs_asterism**tel*wfs*wfs_geom


wfs.set_slopes_units(tomographic_reconstructor = dm.modes@reconstructor, src = ngs_asterism)
wfs_geom.set_slopes_units(tomographic_reconstructor = dm.modes@reconstructor, src = ngs_asterism)

print(wfs.slopes_units)
print(wfs_geom.slopes_units)

#%% Test tomographic reconstruction:
# plt.close('all')

# propagate to geometric SHWFS 
ngs_asterism ** atm *tel*wfs*wfs_geom
# save both geometric and diffractive SHWFS signals
wfs_signal = np.hstack(wfs.signal)
wfs_signal_geo = np.hstack(wfs_geom.signal)

# apply the correction and show the residuals
dm.coefs = -reconstructor@wfs_signal
science ** atm *tel*dm
plt.figure()
plt.imshow(science.phase)
plt.title('Diffractive SHWFS - Residual WFE: ' +str(np.round(np.std(science.phase[science.mask]), 4)) + ' rads')

# apply the correction and show the residuals
dm.coefs = -reconstructor@wfs_signal_geo
science ** atm *tel*dm
plt.figure(),plt.imshow(science.phase)
plt.title('Geometric SHWFS - Residual WFE: ' +str(np.round(np.std(science.phase[science.mask]), 4)) + ' rads')

#%%
displayMap(wfs_geom.signal_2D,axis=0)
plt.colorbar()

displayMap(wfs.signal_2D,axis=0)
plt.colorbar()
#%%
# Interaction matrix using a zonal approach and the geometric SH, with a stroke equal to calib_src.wavelength/2/np.pi
# This matrix is necessary for the pseudo-open loop reconstruction

calib_src = Source('V', 0)
calib_src**tel*wfs_geom

dm_eye = np.eye(dm.nValidAct)
imat = np.zeros((wfs_geom.nValidSubaperture*2, dm.nValidAct))

for i_act in tqdm(range(dm.nValidAct)):

    dm.coefs = dm_eye[:, i_act]*calib_src.wavelength/2/np.pi
    calib_src**tel*dm*wfs_geom

    wfsSignal = np.hstack(wfs_geom.signal)

    imat[:, i_act] = wfsSignal

imat = imat*2*np.pi/calib_src.wavelength
imat = np.vstack([imat]*n_ngs)

plt.imshow(imat)
plt.title("Interaction matrix", fontsize=14,pad=10)
plt.colorbar()
plt.show()


# #%%
# from OOPAO.tools.tools import strehlMeter
# plt.close('all')
# dm.coefs=0

# # To make sure to always replay the same turbulence, generate a new phase screen for the atmosphere and combine it with the Telescope
# atm.generateNewPhaseScreen(seed=10)

# from OOPAO.Detector import Detector
# # science detector
# science_cam = Detector(tel.resolution*4)
# science_cam.psf_sampling = 4
# science_cam.integrationTime = tel.samplingTime*1

# science**tel*science_cam
# science_psf_ref = science_cam.frame.copy()

# # NGS detector
# ngs_cam = Detector(tel.resolution*2)
# ngs_cam.psf_sampling = 4
# ngs_cam.integrationTime = tel.samplingTime

# # Save reference frame
# target**tel*ngs_cam
# ngs_psf_ref = ngs_cam.frame.copy()


# # propagate both sources
# ngs_asterism**atm*tel
# target**atm*tel*ngs_cam
# science**atm*tel*science_cam


# # loop parameters
# n_loop = 500  # number of iterations
# gain_cl = 0.4  # integrator gain
# wfs.cam.photonNoise = False  # enable photon noise on the WFS camera
# display = True  # enable the display
# frame_delay = 1  # number of frame delay

# # variables used to to save closed-loop data data
# SR_ngs = np.zeros(n_loop)
# SR_science = np.zeros(n_loop)

# wfe_atmosphere = np.zeros(n_loop)
# wfe_residual_science = np.zeros(n_loop)
# wfe_residual_NGS = np.zeros(n_loop)
# wfsSignal = np.arange(0, wfs.nSignal*ngs_asterism.n_source)*0  # buffer to simulate the loop delay

# # configure the display pannel
# plot_obj = cl_plot(list_fig=[2*np.pi * (atm.OPD/atm.wavelength),  # list of data for the different subplots
#                               target.phase,
#                               science.phase,
#                               np.block([[wfs.cam.frame[0,:,:],wfs.cam.frame[1,:,:],wfs.cam.frame[2,:,:]]]),
#                               [dm.coordinates[:, 0], np.flip(dm.coordinates[:, 1]), dm.coefs],
#                               [[0, 0], [0, 0], [0, 0]],
#                               np.log10(ngs_cam.frame),
#                               np.log10(science_cam.frame)],
#                     type_fig=['imshow',  # type of figure for the different subplots
#                               'imshow',
#                               'imshow',
#                               'imshow',
#                               'scatter',
#                               'plot',
#                               'imshow',
#                               'imshow'],
#                     list_title=['Turbulence [rads]',  # list of title for the different subplots
#                                 'NGS residual [rads]',
#                                 'science residual [rads]',
#                                 'wfs Detector',
#                                 'DM Commands',
#                                 None,
#                                 None,
#                                 None],
#                     list_legend=[None,  # list of legend labels for the subplots
#                                 None,
#                                 None,
#                                 None,
#                                 None,
#                                 ['science@'+str(science.coordinates[0])+'"', 'NGS@'+str(target.coordinates[0])+'"'],
#                                 None,
#                                 None],
#                     list_label=[None,  # list of axis labels for the subplots
#                                 None,
#                                 None,
#                                 None,
#                                 None,
#                                 ['Time', 'WFE [nm]'],
#                                 ['NGS PSF@' + str(target.coordinates[0]) + '" -- FOV: ' + str(np.round(ngs_cam.fov_arcsec, 2)) + '"', ''],
#                                 ['science PSF@' + str(science.coordinates[0]) + '" -- FOV: ' + str(np.round(science_cam.fov_arcsec, 2)) + '"', '']],
#                     n_subplot=[4, 2],
#                     list_display_axis=[None,  # list of the subplot for which axis are displayed
#                                       None,
#                                       None,
#                                       None,
#                                       None,
#                                       True,
#                                       None,
#                                       None],
#                     list_ratio=[[0.95, 0.95, 0.1],
#                                 [1, 1, 1, 1]],
#                     s=20)  # size of the scatter markers


# for i in range(n_loop):
#     a = time.time()
#     # update phase screens => overwrite tel.OPD and consequently tel.science.phase
#     atm.update()
#     target**atm*tel
#     # save the wave-front error of the incoming turbulence within the pupil
#     wfe_atmosphere[i] = np.std(target.phase[np.where(tel.pupil > 0)])
#     # propagate light from the ngs through the atmosphere, telescope, DM to the wfs and ngs camera
#     target**atm*tel*dm*ngs_cam
#     ngs_asterism**atm*tel*dm*wfs

#     # save residuals corresponding to the ngs
#     wfe_residual_NGS[i] = np.std(target.phase[np.where(tel.pupil > 0)])
#     # save Strehl ratio from the PSF image
#     SR_ngs[i] = strehlMeter(PSF=ngs_cam.frame, tel=tel, PSF_ref=ngs_psf_ref, display=False)
#     # save the OPD seen by the ngs
#     PHASE_NGS = target.phase.copy()
#     if display:
#         NGS_PSF = np.log10(np.abs(ngs_cam.frame))

#     # propagate light from the science through the atmosphere, telescope, DM to the science camera
#     science**atm*tel*dm*science_cam
#     # save residuals corresponding to the science
#     wfe_residual_science[i] = np.std(science.phase[np.where(tel.pupil > 0)])
#     # save the OPD seen by the science
#     PHASE_science = science.phase.copy()
#     # save Strehl ratio from the PSF image
#     SR_science[i] = strehlMeter(PSF=science_cam.frame, tel=tel, PSF_ref=science_psf_ref, display=False)

#     # store the slopes after propagating to the wfs <=> 1 frames delay
#     if frame_delay == 1:
#         wfsSignal = np.hstack(wfs.signal)
#         wfsSignal = wfsSignal-imat@dm.coefs


#     # apply the commands on the DM (POLC)
#     dm.coefs = (1-gain_cl)*dm.coefs - gain_cl * (reconstructor@wfsSignal)

#     # store the slopes after computing the commands <=> 2 frames delay
#     if frame_delay == 2:
#         wfsSignal = np.hstack(wfs.signal)
#         wfsSignal = wfsSignal-imat@dm.coefs

#     # print('Elapsed time: ' + str(time.time()-a) + ' s')

#     # update displays if required
#     if display and i > 1:
#         science_PSF = np.log10(np.abs(science_cam.frame))
#         # update range for PSF images
#         plot_obj.list_lim = [None,
#                               None,
#                               None,
#                               None,
#                               None,
#                               None,
#                               [NGS_PSF.max()-6, NGS_PSF.max()],
#                               [science_PSF.max()-6, science_PSF.max()]]
#         # update title
#         plot_obj.list_title = ['Turbulence '+str(np.round(wfe_atmosphere[i], 4))+'[rads]',
#                                 'NGS residual '+str(np.round(wfe_residual_NGS[i], 4))+'[rads]',
#                                 'science residual '+str(np.round(wfe_residual_science[i], 4))+'[rads]',
#                                 'wfs Detector',
#                                 'DM Commands',
#                                 None,
#                                 None,
#                                 None]

#         cl_plot(list_fig=[2*np.pi * (atm.OPD/atm.wavelength),
#                           PHASE_NGS,
#                           PHASE_science,
#                           np.block([[wfs.cam.frame[0,:,:],wfs.cam.frame[1,:,:],wfs.cam.frame[2,:,:]]]),
#                           dm.coefs,
#                           [np.arange(i+1), wfe_residual_science[:i+1], wfe_residual_NGS[:i+1]],
#                           NGS_PSF,
#                           science_PSF],
#                 plt_obj=plot_obj)
#         plt.pause(0.01)
#         if plot_obj.keep_going is False:
#             break
#     print('-----------------------------------')
#     print('Loop'+str(i) + '/' + str(n_loop))
#     print('NGS: Strehl ratio [%] : ', np.round(SR_ngs[i],1), ' WFE [rads] : ', np.round(wfe_residual_NGS[i],4))
#     print('science: Strehl ratio [%] : ', np.round(SR_science[i],1), ' WFE [rads] : ', np.round(wfe_residual_science[i],4))



# #%%
# plt.figure()

# plt.plot(wfe_residual_NGS,label='Residuals')
# plt.plot(wfe_atmosphere,label="Total")
# plt.title('Residuals',fontsize=14,pad=10)
# plt.ylabel('Residual [nm rms]',fontsize=12,labelpad=10)
# plt.legend(fontsize=12)
# plt.grid()
# plt.show()


# plt.figure()
# plt.plot(SR_ngs)
# plt.title("Strehl ratio",fontsize=14,pad=10)
# plt.grid()
# plt.show()