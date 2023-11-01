from sgp4.api import Satrec

from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation, ITRS, AltAz
from astropy import units as u

class Satellite():
    def __init__(self, tle):
        self.__data = Satrec.twoline2rv(tle[0], tle[1])

    def at(self, t, obsloc=None):
        e, p, v = self.__data.sgp4(t.jd1, t.jd2)
        print(e, p, v)

        p = CartesianRepresentation(p * u.km)
        v = CartesianDifferential(v * u.km/u.s)

        self.TEME = TEME(p.with_differentials(v), obstime=t)

        if obsloc is not None:
            itrs_geo = self.TEME.transform_to(ITRS(obstime=t))
            itrs_geo_p = itrs_geo.cartesian.without_differentials()
            itrs_geo_v = itrs_geo.cartesian.differentials['s']
            topo_itrs_p = itrs_geo_p - obsloc.get_itrs(t).cartesian
            topo_itrs_repr = topo_itrs_p.with_differentials(itrs_geo_v)
            itrs_topo = ITRS(topo_itrs_repr, obstime=t, location=obsloc)
            self.aa = itrs_topo.transform_to(AltAz(obstime=t, location=obsloc))
            print(self.aa)
        else:
            self.aa = None
        
        return self


