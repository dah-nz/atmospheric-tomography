import matplotlib.pyplot as plt

if __name__ == "__main__":


    # from tomography import test_phase_screens
    # test_phase_screens.run_tests()

    # from tomography import test_oopao
    # test_oopao.run_tests()

    # from tomography import test_regime_data
    # test_regime_data.run_tests()

    # from tomography import test_asterism_quality
    # test_asterism_quality.run_tests()

    # In here are where the asterism quality and regime models are trained
    # from tomography import test_asterisms
    # test_asterisms.run_tests()

    # from tomography import test_tomography
    # test_tomography.run_tests()

    from ml import test_models

    test_models.run_models_over_different_heuristics()
    test_models.display_models_over_different_heuristics()
    test_models.run_models_over_different_Dr0()
    test_models.display_models_over_different_Dr0()

    test_models.run_and_display_single_example()

    plt.show()

    quit()