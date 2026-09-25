import numpy as np

def config(a, b, c):
    return f'{{\n\
      "directory": "D:/Data/2026/altitudes2/altitudes_{int(a*1000)}_{int(b*1000)}_{int(c*1000)}",\n\
      "num_modes": 36,\n\
      "aperture_size": 100,\n\
      "diameter": 1.8,\n\
      "r0": [0.064677, 0.497862, 0.257559],\n\
      "layers": 3,\n\
      "altitudes": [{int(a*1000)},{int(b*1000)},{int(c*1000)}],\n\
      "magnitude": 14,\n\
      "train_models": ["GuideStarEmbed_baseline",\n\
                       "GuideStarEmbed_pseudo_lsq",\n\
                       "GuideStarEmbed_pseudo_mmse",\n\
                       "GuideStarEmbed_NoInverseKzz",\n\
                       "ResNetEmbed"\n\
                    ],\n\
      "lsq_off": false,\n\
      "glao_off": false,\n\
      "ao_off": false,\n\
      "mmse_off": false,\n\
      "sample_set_off": true,\n\
      "sample_seed": 123,\n\
      "epochs": 0,\n\
      "training_size": 0,\n\
      "verification_size": 1000,\n\
      "save_frequency": 200,\n\
      "display_training": false,\n\
      "shwfs": false,\n\
      "noise": false,\n\
      "asterism_size": 3,\n\
      "angle_threshold": 5,\n\
      "satellites": false,\n\
      "period": 72,\n\
      "load_previous": false,\n\
      "continue_training": false,\n\
      "verification_skip_lsq": true,\n\
      "asterism_model_location": "C:/Code/PhD/Data/2026/Models",\n\
      "quality_model": "disabled",\n\
      "regime_model": "disabled",\n\
      "init_model_directory": "D:/Data/2026/all_asterisms/20260817_143646"\n\
    }}'

with open('output.txt', 'w') as f:
    first = True
    A = np.linspace(0, 2, num=21, endpoint=True)
    B = np.linspace(2, 8, num=25, endpoint=True)
    C = np.linspace(7, 14, num=29, endpoint=True)

    print(A,B,C)

    f.write('[')

    for a in A:
        if not first:
            f.write(',')
        else:
            first = False
        f.write(config(a, 5, 10))

    for b in B:
        if b != 5:
            f.write(',')
            f.write(config(1, b, 10))

    for c in C:
        if c != 10:
            f.write(',')
            f.write(config(1, 5, c))

    f.write(']')
