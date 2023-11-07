import datetime as dt

from sgp4.api import Satrec

from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation, ITRS, AltAz
from astropy import units as u

from skyfield.api import load


DEFAULT_CATALOG_URL = 'http://celestrak.org/NORAD/elements/visual.txt'

def get_satellite_catalog(url=None):
    """ Retrieve catalog of skyfield satellite objects from url """
    if url == None:
        url = DEFAULT_CATALOG_URL
    return load.tle_file(url) 

def find_visible_satellites(location, satellites=None, time=None, tdelta=24, sat_horizon=0, sun_horizon=0, url=None, eph=None):
    """ Get a list of visible satellites, with times they are in sunlight.
    Checks for sun and moon position, so no satellite passes during the day are returns, nor are satellites that are transitting the moon

    Returns a list of (Satellite, time) tuples, where time is a list of tuples giving the UTC times the satellite appears then disappears
    Inputs:
    location    - Location object giving position of observer
    satellites  - catalog of satellite objects
    time        - datetime object, giving the start time when looking for satellites begins
    tdelta      - time in hours from start time to search for visible satellites
    tz          - timezone for computation
    sat_horizon - Altitude angle where satellites above this angle are deemed potentially visible (0 is the actual horizon from location)
    sun_horizon - Altitude angle where the sun is below the horizon (0 is the actual horizon from location)
    url         - url of satellite catalog - only used if satellites is not defined
    eph         - planetary ephemeris data for determining sun, earth and moon locations
    """
    # Get time to find satellites
    ts = load.timescale()
    if time == None:
        time = dt.datetime.now(tz=dt.timezone.utc)
    t0 = ts.from_datetime(t)
    t1 = ts.from_datetime(t + dt.timedelta(hours=tdelta))

    # Load satellites from URL
    if satellites == None:
        satellites = get_satellite_catalog(url)
    print('Loaded', len(satellites), 'satellites')

    # Load planet and sun ephemeris data
    if eph == None:
        eph = load('de421.bsp')

    event_names = ['Rise above horizon', 'Culminate', 'Set below horizon']

    pos = location.wgs84_latlon()

    visible_satellites = []

    for satellite in satellites:
        t, events = satellite.find_events(pos, t0, t1, altitude_degrees=sat_horizon)
        print(satellite)
        diff = satellite - pos

        times = []

        for ti, event in zip(t, events):
            l = diff.at(ti)
            s = location.get_sun_altitude(eph, ti)

            name = event_names[event]
            if s < sun_horizon:   # sun under the horizon
                if name == 'Culminate':
                    name += ' at {}'.format(l.altaz()[0])
                tn = ti.utc_datetime()
                print("{}: {} ({})".format(
                    tn.strftime('%Y %b %d %H:%M:%S'),
                    name,
                    'sunlit' if l.is_sunlit(eph) else 'dark'
                ))

class Satellite():
    def __init__(self, tle):
        self.__data = Satrec.twoline2rv(tle[0], tle[1])

    def at(self, t, obsloc=None):
        e, p, v = self.__data.sgp4(t.jd1, t.jd2)

        if e != 0:
            print(e)
            quit()

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
        else:
            self.aa = None
        
        return self


if __name__ == "__main__":
    import telescope
    t = telescope.Telescope(1.0, 6.5, telescope.UC_MT_JOHN_OBS)
    find_visible_satellites(t)