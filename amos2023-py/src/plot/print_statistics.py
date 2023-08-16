import numpy as np

coeffs = [16] # [16, 64]
snr = [0] # [0, 3, 10, 20, 30]

for n in coeffs:
    for noise in snr:
        with open("./results/results_residuals_{}_SNR_{}.data".format(n, noise), "r") as f:
            raw_data = f.readlines()

            keys = raw_data[0].strip().split(',')
            keys = [k.strip() for k in keys]

            for l in raw_data[1:]:
                data.append(l.strip().split(','))

            angle = set()
            asterism = set()
            basis = set()

            for d in data:
                if int(d[keys.index('N Coeffs')]) == n:
                    angle.add(int(d[keys.index('Target Gamma')]))
                    asterism.add(int(d[keys.index('N Guide Stars')]))
                    basis.add(d[keys.index('Basis Function')])
            
            angle = sorted(list(angle))
            asterism = sorted(list(asterism))
            basis = list(basis)

            for b in basis:
                rmse_i = {}     # RMSE (Recon - Ideal)
                rmse_r = {}     # RMSE (Recon - Actual)
                rmse_a = {}     # RMSE (Ideal - Actual)
                rmse_b = {}     # RMSE (Baseline - Actual)

                mssim = {}      # MSSIM (Recon and Ideal Residual PSFs)

                strehl = {}     # Strehl (Between Recon and Ideal Residual PSFs)

                fwhm_i = {}     # FWHM (Ideal Residual PSF)
                fwhm_r = {}     # FWHM (Recon Residual PSF)

                for d in [rmse_i, rmse_r, rmse_b, mssim, strehl, fwhm_i, fwhm_r]:
                    for g in angle:
                        d[(g, -1)] = []
                        for a in asterism:
                            d[(g, a)] = []

                iNG = keys.index('N Guide Stars')
                iTG = keys.index('Target Gamma')
                iB = keys.index('Basis Function')

                iEI = keys.index('RMSE I:R')
                iER = keys.index('RMSE R:A')
                iEA = keys.index('RMSE I:A')
                iEB = keys.index('RMSE B:A')

                iMS = keys.index('MSSIM R:A-I:A')

                iIR = keys.index('Max Int. R:A')
                iII = keys.index('Max Int. I:A')

                iFI = keys.index('FWHM I:A')
                iFR = keys.index('FWHM R:A')

                for d in data:
                    if d[iB] == b:
                        rmse_i[(d[iNG], d[iTG])].append(d[iEI])         # RMSE (Recon - Ideal)
                        rmse_r[(d[iNG], d[iTG])].append(d[iER])         # RMSE (Recon - Actual)
                        rmse_a[(d[iNG], d[iTG])].append(d[iEA])         # RMSE (Ideal - Actual)
                        rmse_b[(d[iNG], d[iTG])].append(d[iEB])         # RMSE (Baseline - Actual)
                        mssim [(d[iNG], d[iTG])].append(d[iMS])         # MSSIM (Recon and Ideal Residual PSFs)
                        strehl[(d[iNG], d[iTG])].append(d[iIR]/d[iII])  # Strehl (Between Recon and Ideal Residual PSFs)
                        fwhm_i[(d[iNG], d[iTG])].append(d[iFI])         # FWHM (Ideal Residual PSF)
                        fwhm_r[(d[iNG], d[iTG])].append(d[iFR])         # FWHM (Recon Residual PSF)

                        rmse_i[(d[iNG], -1)].append(d[iEI])             # RMSE (Recon - Ideal)
                        rmse_i[(d[iNG], -1)].append(d[iEI])             # RMSE (Recon - Ideal)
                        rmse_r[(d[iNG], -1)].append(d[iER])             # RMSE (Recon - Actual)
                        rmse_a[(d[iNG], -1)].append(d[iEA])             # RMSE (Ideal - Actual)
                        rmse_b[(d[iNG], -1)].append(d[iEB])             # RMSE (Baseline - Actual)
                        mssim [(d[iNG], -1)].append(d[iMS])             # MSSIM (Recon and Ideal Residual PSFs)
                        strehl[(d[iNG], -1)].append(d[iIR]/d[iII])      # Strehl (Between Recon and Ideal Residual PSFs)
                        fwhm_i[(d[iNG], -1)].append(d[iFI])             # FWHM (Ideal Residual PSF)
                        fwhm_r[(d[iNG], -1)].append(d[iFR])             # FWHM (Recon Residual PSF)

                for n in asterism + [-1]:
                    for g in angle:
                        print("RMSE RI MEAN: {}".format(np.mean(rmse_i[(g, n)])))
                        print("RMSE RI STD: {}".format(np.mean(rmse_i[(g, n)])))
                        print("RMSE RI: {}".format(np.mean(rmse_i[(g, n)])))
                

                    
