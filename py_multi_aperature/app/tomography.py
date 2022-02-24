from parameters import parameters
from simulation import Sim
import tomo_covariance
import tomo_projection
import plot


def main():

    p = parameters()
    sim = Sim(p)

    t_covar = tomo_covariance.covariance(sim)
    plot.plot_tomography("Tomography - Covariance", sim, t_covar)

    t_proj = tomo_projection.projection(sim)
    plot.plot_tomography("Tomography - Projection", sim, t_proj, final=True)

    plot.show()


if __name__ == "__main__":
    main()