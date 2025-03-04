# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:01:52 2024

@author: cheritier
"""


import time

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix #FromPhaseScreen
from OOPAO.tools.displayTools import cl_plot, displayMap
import matplotlib.gridspec as gridspec


# %%
# plt.ion()
# number of subaperture for the WFS
n_subaperture = 20
pps = 8


#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = pps*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 8,                                      # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.0,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 10 )                                      # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

# Apply spiders to the telescope pupil
# thickness_spider    = 0.05                                                       # thickness of the spiders in m
# angle               = [45, 135, 225, 315]                                        # angle in degrees for each spider
# offset_Y            = [-0.2, -0.2, 0.2, 0.2]                                     # shift offsets for each spider
# thickness_spider    = 0.1                                                # thickness of the spiders in m
# angle               = [30, 150, 210, 330]                                        # angle in degrees for each spider
# offset_Y            = [-0.1, -0.1, 0.1, 0.1]                                     # shift offsets for each spider
# offset_X            = None

# tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

# display current pupil
plt.figure()
plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel

# create the Scientific Target object located at 10 arcsec from the  ngs
src = Source(optBand     = 'K',           # Optical band (see photometry.py)
             magnitude   = 8,              # Source Magnitude
             coordinates = [1,0])        # Source coordinated [arcsec,deg]

# combine the SRC to the telescope using '*'
src*tel

# check that the ngs and tel.src objects are the same
tel.src.print_properties()

# compute PSF 
tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,4])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

#%% -----------------------     ATMOSPHERE   ----------------------------------
from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.15,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]


# atm = Atmosphere(telescope     = tel,                               # Telescope                              
#                  r0            = 0.15,                              # Fried Parameter [m]
#                  L0            = 25,                                # Outer Scale [m]
#                  fractionalR0  = [1   ], # Cn2 Profile
#                  windSpeed     = [10    ], # Wind Speed in [m]
#                  windDirection = [0   ], # Wind Direction in [degrees]
#                  altitude      = [50000 ]) # Altitude Layers in [m]

# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel)

# The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
atm.update()

# display the atm.OPD = resulting OPD 
plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

# display the atmosphere layers for the sources specified in list_src: 
atm.display_atm_layers(list_src=[ngs,src])

# the sources coordinates can be updated on the fly: 
src.coordinates = [0,0]
atm.display_atm_layers(list_src=[ngs,src])


#%% -----------------------     Scientific Detector   ----------------------------------
from OOPAO.Detector import Detector

# define a detector with its properties (see Detector class for further documentation)
cam = Detector(integrationTime = tel.samplingTime,      # integration time of the detector
               photonNoise     = True,                  # enable photon noise
               readoutNoise    = 0,                     # readout of the detector in [e-/pixel]
               QE              = 1,                   # quantum efficiency
               psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
               binning         = 1)                     # Binning factor of the PSF

cam_binned = Detector( integrationTime = tel.samplingTime,      # integration time of the detector
                       photonNoise     = True,                  # enable photon noise
                       readoutNoise    = 2,                     # readout of the detector in [e-/pixel]
                       QE              = 0.8,                   # quantum efficiency
                       psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
                       binning         = 4)                     # Binning factor of the PSF


# computation of a PSF on the detector using the '*' operator
src*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

#%%         PROPAGATE THE LIGHT THROUGH THE ATMOSPHERE
# The Telescope and Atmosphere can be combined using the '+' operator (Propagation through the atmosphere): 
tel+atm # This operations makes that the tel.OPD is automatically over-written by the value of atm.OPD when atm.OPD is updated. 

# It is possible to print the optical path: 
tel.print_optical_path()

# computation of a PSF on the detector using the '*' operator
atm*ngs*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

# The Telescope and Atmosphere can be separated using the '-' operator (Free space propagation) 
tel-atm
tel.print_optical_path()

# computation of a PSF on the detector using the '*' operator
ngs*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

# mis-registrations object (rotation, shifts..)
misReg = MisRegistration()
misReg.shiftX = 0           # in [m]
misReg.shiftY = 0           # in [m]
misReg.rotationAngle = 0    # in [deg]


# specifying a given number of actuators along the diameter: 
nAct = n_subaperture+1
    
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = nAct-1,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    misReg       = misReg,                     # Mis-registration associated 
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = tel.D/nAct) #,floating_precision=32)                 # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    

# plot the dm actuators coordinates with respect to the pupil
plt.figure()
plt.imshow(np.reshape(np.sum(dm.modes**5,axis=1),[tel.resolution,tel.resolution]).T + tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')


#%% -----------------------     SHACK-HARTMANN WFS   ----------------------------------
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Pyramid import Pyramid

# make sure tel and atm are separated to initialize the PWFS
tel.isPaired = False
tel.resetOPD()

sh_wfs = ShackHartmann( nSubap           = n_subaperture,
                        telescope        = tel,
                        lightRatio       = 0.5,
                        shannon_sampling = True)

p_wfs = Pyramid(nSubap            = n_subaperture,
                telescope         = tel,
                lightRatio        = 0.5,
                modulation        = 3,
                binning           = 1,
                n_pix_separation  = 4,
                n_pix_edge        = 2,
                postProcessing    = 'slopesMaps_incidence_flux')

for t_wfs in [p_wfs, sh_wfs]:
    # propagate the light to the Wave-Front Sensor
    tel*t_wfs

    plt.close('all')
    plt.figure()
    plt.imshow(t_wfs.cam.frame)
    plt.title('WFS Camera Frame')

    plt.figure()
    plt.imshow(t_wfs.signal_2D)
    plt.title('WFS Signal')

    # The photon Noise of the detector can be disabled the same way than for a Detector class
    t_wfs.cam.photonNoise = False

    ngs*tel*t_wfs

    plt.figure()
    plt.imshow(t_wfs.cam.frame)
    plt.title('WFS Camera Frame - Without Noise')

    t_wfs.cam.photonNoise = True
    ngs*tel*t_wfs
    plt.figure()
    plt.imshow(t_wfs.cam.frame)
    plt.title('WFS Camera Frame - With Noise')


#%% -----------------------     Modal Basis - Zernike  ----------------------------------
from OOPAO.Zernike import Zernike

#% ZERNIKE Polynomials
# create Zernike Object
Z = Zernike(tel, 120)
# compute polynomials for given telescope
Z.computeZernike(tel)

# mode to command matrix to project Zernike Polynomials on DM
M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes

# show the first 10 zernikes applied on the DM
dm.coefs = M2C_zernike[:,:10]
tel*dm
displayMap(tel.OPD)

#%% -----------------------     Modal Basis - KL Basis  ----------------------------------


# from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
# # # use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
# M2C_KL = compute_KL_basis(tel, atm, dm,lim = 1e-2) # matrix to apply modes on the DM

# # apply the 10 first KL modes
# dm.coefs = M2C_KL[:,:10]
# # propagate through the DM
# ngs*tel*dm
# # show the first 10 KL modes applied on the DM
# displayMap(tel.OPD)
#%% -----------------------     Calibration: Interaction Matrix  ----------------------------------

# amplitude of the modes in m
stroke=ngs.wavelength/16 #1e-9
# zonal Interaction Matrix
M2C_zonal = np.eye(dm.nValidAct)

# modal Interaction Matrix for 300 modes
M2C_modal = M2C_zernike[:,:20]
# M2C_modal = M2C_KL[:,:300]

tel-atm

from basis import zernike
z = zernike.Zernike(21)

# phase_screens = []
# for i in range(20):
#     c = np.zeros(21)
#     c[i+1] = 1.0
#     phase_screens.append(z.compose(pps*n_subaperture, c))

# phase_screens = np.array(phase_screens).T

from tomography import phase_screen

gen = phase_screen.PhaseScreenGenerator(pps*n_subaperture, 1.8, r0=0.07)
screen = gen.get()
screen -= np.mean(screen, where=z.out((pps * n_subaperture, pps * n_subaperture), 0)>0)

sh_calib_modal = InteractionMatrix( ngs            = ngs,
                                    atm            = atm,
                                    tel            = tel,
                                    dm             = dm,
                                    wfs            = sh_wfs,   
                                    M2C            = M2C_modal, # M2C matrix used 
                                    stroke         = stroke,    # stroke for the push/pull in M2C units
                                    nMeasurements  = 1,         # number of simultaneous measurements
                                    noise          = 'off',     # disable wfs.cam noise 
                                    display        = True,      # display the time using tqdm
                                    single_pass    = True)      # only push to compute the interaction matrix instead of push-pull

p_calib_modal = InteractionMatrix(  ngs            = ngs,
                                    atm            = atm,
                                    tel            = tel,
                                    dm             = dm,
                                    wfs            = p_wfs,   
                                    M2C            = M2C_modal, # M2C matrix used 
                                    stroke         = stroke,    # stroke for the push/pull in M2C units
                                    nMeasurements  = 1,         # number of simultaneous measurements
                                    noise          = 'off',     # disable wfs.cam noise 
                                    display        = True,      # display the time using tqdm
                                    single_pass    = True)      # only push to compute the interaction matrix instead of push-pull
# zonal interaction matrix
for t_wfs, t_calib_modal in [(p_wfs, p_calib_modal), (sh_wfs, sh_calib_modal)]:

    # calib_modal = InteractionMatrixFromPhaseScreen(ngs,atm,tel,wfs,phase_screens,stroke,phaseOffset=0,nMeasurements=1,noise='off',print_time=True)

    plt.figure()
    plt.plot(np.std(t_calib_modal.D,axis=0))
    plt.xlabel('Mode Number')
    plt.ylabel('WFS slopes STD')

    #%% Define instrument and WFS path detectors
    from OOPAO.Detector import Detector

    # instrument path
    src_cam = Detector(tel.resolution*4)
    src_cam.psf_sampling = 4
    src_cam.integrationTime = tel.samplingTime*1
    # put the scientific target off-axis to simulate anisoplanetism (set to  [0,0] to remove anisoplanetism)
    src.coordinates = [0.4,0]

    # WFS path
    ngs_cam = Detector(tel.resolution)
    ngs_cam.psf_sampling = 4
    ngs_cam.integrationTime = tel.samplingTime

    # initialize Telescope DM commands
    tel.resetOPD()
    dm.coefs=0
    ngs*tel*dm*t_wfs
    if t_wfs is p_wfs:
        t_wfs*t_wfs.focal_plane_camera
    # Update the r0 parameter, generate a new phase screen for the atmosphere and combine it with the Telescope
    # atm.r0 = 0.15
    atm.generateNewPhaseScreen(seed = 10)


    # opd = screen / (2 * np.pi / src.wavelength)

    atm.set_OPD(screen)

    print(Z.modes.shape)
    print(screen.shape)

    z_coeffs = [np.sum(screen.ravel()[tel.pupilLogical] * Z.modes[:, n], dtype=np.float64)/np.sum(Z.modes[:, n] * Z.modes[:, n]) for n in range(20)]
    z_coeffs_mine = z.decompose(screen)[1:]

    t_wfs.wfs_measure(screen)

    if t_wfs is sh_wfs:
        sh_coeffs = np.matmul(t_calib_modal.M, t_wfs.signal)
    else:
        p_coeffs = np.matmul(t_calib_modal.M, t_wfs.signal)

plt.figure()
plt.plot(z_coeffs, label="Dot Zernike (OOPAO)")
plt.plot(z_coeffs_mine, label="Dot Zernike (van Dam)")
plt.plot(sh_coeffs * (2 * np.pi / src.wavelength), label="SH Zernike (OOPAO)")
plt.plot(p_coeffs * (2 * np.pi / src.wavelength), label="Pyramid Zernike (OOPAO)")
plt.legend()

plt.figure()
plt.plot(z_coeffs / np.max(z_coeffs), label="Dot Zernike (OOPAO)")
plt.plot(z_coeffs_mine / np.max(z_coeffs_mine), label="Dot Zernike (van Dam)")
plt.plot(sh_coeffs /np.max(sh_coeffs), label="SH Zernike")
plt.plot(p_coeffs /np.max(p_coeffs), label="Pyramid Zernike")
plt.legend()



# plt.show()

tel+atm

tel.computePSF(4)
# plt.close('all')
    
# These are the calibration data used to close the loop
# calib_CL    = calib_modal
# M2C_CL      = M2C_modal


# combine telescope with atmosphere
tel+atm

# initialize DM commands
atm*ngs*tel*ngs_cam
atm*src*tel*src_cam


# plt.show()

nLoop = 500
# allocate memory to save data
SR_NGS                      = np.zeros(nLoop)
SR_SRC                      = np.zeros(nLoop)
total                   = np.zeros(nLoop)
residual_SRC                = np.zeros(nLoop)
residual_NGS                = np.zeros(nLoop)

# wfsSignal               = np.arange(0,wfs.nSignal)*0

# plot_obj = cl_plot(list_fig          = [atm.OPD,
#                                         tel.mean_removed_OPD,
#                                         tel.mean_removed_OPD,
#                                         [[0,0],[0,0],[0,0]],
#                                         wfs.cam.frame,
#                                         # [dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],
#                                         wfs.focal_plane_camera.frame,
#                                         np.log10(tel.PSF),
#                                         np.log10(tel.PSF)],
#                    type_fig          = ['imshow',
#                                         'imshow',
#                                         'imshow',
#                                         'plot',
#                                         'imshow',
#                                         'imshow',
#                                         'imshow',
#                                         'imshow'],
#                    list_title        = ['Turbulence [nm]',
#                                         'NGS@'+str(ngs.coordinates[0])+'" WFE [nm]',
#                                         'SRC@'+str(src.coordinates[0])+'" WFE [nm]',
#                                         None,
#                                         'WFS Detector',
#                                         'WFS Focal Plane Camera',
#                                         None,
#                                         None],
#                    list_legend       = [None,None,None,['SRC@'+str(src.coordinates[0])+'"','NGS@'+str(ngs.coordinates[0])+'"'],None,None,None,None],
#                    list_label        = [None,None,None,['Time','WFE [nm]'],None,None,['NGS PSF@'+str(ngs.coordinates[0])+'" -- FOV: '+str(np.round(ngs_cam.fov_arcsec,2)) +'"',''],['SRC PSF@'+str(src.coordinates[0])+'" -- FOV: '+str(np.round(src_cam.fov_arcsec,2)) +'"','']],
#                    n_subplot         = [4,2],
#                    list_display_axis = [None,None,None,True,None,None,None,None],
#                    list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
# loop parameters

sh_gain = {}
p_gain = {}

for noise in ["noise", "ideal"]:
    gainCL                  = 0.0
    sh_wfs.cam.photonNoise  = noise == "noise"
    p_wfs.cam.photonNoise   = noise == "noise"
    display                 = True
    frame_delay             = 1
    # reconstructor = M2C_CL@calib_CL.M

        # tel.resetOPD()
        # dm.coefs=0
        # atm*ngs*tel*dm*t_wfs
        # t_wfs*t_wfs.focal_plane_camera

        # z_coeffs = [np.sum(screen.ravel()[tel.pupilLogical] * Z.modes[:, n], dtype=np.float64)/np.sum(Z.modes[:, n] * Z.modes[:, n]) for n in range(20)]
        # p_coeffs = np.matmul(p_calib_modal.M, p_wfs.signal)
        # sh_coeffs = np.matmul(sh_calib_modal.M, sh_wfs.signal)

    sh_gain[noise] = []
    p_gain[noise] = []

    for i in range(nLoop):
        a=time.time()
        # update phase screens => overwrite tel.OPD and consequently tel.src.phase
        atm.generateNewPhaseScreen()
        # atm.update()
        phase = atm.OPD * (2 * np.pi / src.wavelength)
        z_coefs = z.decompose(phase)
        # save phase variance
        total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
        # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied
        atm*ngs*tel*dm*sh_wfs*ngs_cam

        sh_coeffs = np.matmul(sh_calib_modal.M, sh_wfs.signal)
        
        atm*ngs*tel*dm*p_wfs*ngs_cam
        p_wfs*p_wfs.focal_plane_camera

        p_coeffs = np.matmul(p_calib_modal.M, p_wfs.signal)

        p_gain[noise].append(np.max(p_coeffs) / np.max(z_coefs) * (2 * np.pi / src.wavelength))
        sh_gain[noise].append(np.max(sh_coeffs) / np.max(z_coefs) * (2 * np.pi / src.wavelength))


    # save residuals corresponding to the NGS
    # residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # OPD_NGS         = tel.mean_removed_OPD.copy()

    # if display==True:        
    #     NGS_PSF = np.log10(np.abs(ngs_cam.frame))
    
    # propagate light from the SRC through the atmosphere, telescope, DM to the Instrument camera
    # atm*src*tel*dm*src_cam
    
    # save residuals corresponding to the NGS
    # residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # OPD_SRC         = tel.mean_removed_OPD.copy()
    # if frame_delay ==1:        
    #     wfsSignal=wfs.signal
    
    # apply the commands on the DM
    # print(f"reconstructor: {reconstructor.shape}; wfsSignal: {wfsSignal.shape}; dm.coefs: {dm.coefs.shape}")
    # dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
        
    # store the slopes after computing the commands => 2 frames delay
    # if frame_delay ==2:        
    #     wfsSignal=wfs.signal
    
    # print('Elapsed time: ' + str(time.time()-a) +' s')
    
    # # update displays if required
    # if display==True and i>0:        
        
    #     SRC_PSF = np.log10(np.abs(src_cam.frame))
    #     # update range for PSF images
    #     plot_obj.list_lim = [None,None,None,None,None,None,[NGS_PSF.max()-3, NGS_PSF.max()],[SRC_PSF.max()-4, SRC_PSF.max()]]        
    #     # update title
    #     plot_obj.list_title = ['Turbulence WFE:'+str(np.round(total[i]))+'[nm]',
    #                            'NGS@'+str(ngs.coordinates[0])+'" WFE:'+str(np.round(residual_NGS[i]))+'[nm]',
    #                            'SRC@'+str(src.coordinates[0])+'" WFE:'+str(np.round(residual_SRC[i]))+'[nm]',
    #                             None,
    #                             'WFS Detector',
    #                             'WFS Focal Plane Camera',
    #                             None,
    #                             None]

    #     cl_plot(list_fig   = [1e9*atm.OPD,1e9*OPD_NGS,1e9*OPD_SRC,[np.arange(i+1),residual_SRC[:i+1],residual_NGS[:i+1]],wfs.cam.frame,wfs.focal_plane_camera.frame,NGS_PSF, SRC_PSF],
    #                            plt_obj = plot_obj)
    #     plt.pause(0.001)
    #     if plot_obj.keep_going is False:
    #         break
    # print('Loop'+str(i)+'/'+str(nLoop)+' NGS: '+str(residual_NGS[i])+' -- SRC:' +str(residual_SRC[i])+ '\n')

plt.figure()
plt.hist([sh_gain["noise"], sh_gain["ideal"], p_gain["noise"], p_gain["ideal"]], bins=30, label=["SH Gain (Noisy)", "SH Gain", "P Gain (Noisy)", "P Gain"], density=True, histtype="step")
plt.legend()

plt.show()

#%%