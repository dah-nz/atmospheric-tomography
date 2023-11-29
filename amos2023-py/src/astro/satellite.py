import datetime as dt

from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation, ITRS, AltAz
import astropy.units as u
import astropy.time as aTime

from skyfield.api import load, EarthSatellite, utc

DEFAULT_CATALOG_URL = 'http://celestrak.org/NORAD/elements/visual.txt'

def get_satellite_catalog(url=None):
    """ Retrieve catalog of skyfield satellite objects from url """
    if url == None:
        url = DEFAULT_CATALOG_URL
    satellites = load.tle_file(url)
    for s in satellites:
        s.__class__ = Satellite
    return satellites

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
    t0 = ts.from_datetime(time)
    t1 = ts.from_datetime(time + dt.timedelta(hours=tdelta))

    # Load satellites from URL
    if satellites == None:
        satellites = get_satellite_catalog(url)
    print('Loaded', len(satellites), 'satellites')

    # Load planet and sun ephemeris data
    if eph == None:
        eph = load('de421.bsp')

    event_names = ['Rise above horizon', 'Culminate', 'Set below horizon']

    pos = location.wgs84_latlon

    visible_satellites = []

    for satellite in satellites:
        t, events = satellite.find_events(pos, t0, t1, altitude_degrees=sat_horizon)
        print(satellite)
        diff = satellite - pos

        vis_times = []

        for ti, event in zip(t, events):
            s = location.get_sun_altitude(eph, ti)  # sun position at time

            if s < sun_horizon:   # sun under the horizon
                if event == 0:   # rising
                    ts = ti
                if event == 2:   # setting
                    te = ti
                    vis_times = vis_times + find_visible_times_between(ts, te, diff, eph)

        if len(vis_times) > 0:
            visible_satellites += [(satellite, vt) for vt in vis_times]
            
    return visible_satellites

def find_visible_times_between(t0, t1, pos, eph):
    times = []
    t = t0
    while t1 - t > 0:
        times.append(t)
        t += dt.timedelta(seconds=1)
    sunlit = [pos.at(t).is_sunlit(eph) for t in times]
    if any(sunlit):
        t = []
        ts = times[0]
        for t0, t1, s0, s1 in zip([0] + times, times + [0], [False] + sunlit, sunlit + [False]):
            if not s0 and s1:
                ts = t1
            elif s0 and not s1:
                t.append((ts, t0))
        return t
    return []

    
class Satellite(EarthSatellite):
    def altaz_from(self, location, t):
        if type(t) == aTime.core.Time:
            ts = load.timescale()
            time = t.datetime
            time = time.replace(tzinfo=utc)
            t = ts.from_datetime(time)
        p, v, m = self._position_and_velocity_TEME_km(t)

        if m != None:
            raise RuntimeError(m)

        p = CartesianRepresentation(p * u.km)
        v = CartesianDifferential(v * u.km/u.s)

        t = aTime.Time(t.astimezone(dt.timezone.utc))

        teme = TEME(p.with_differentials(v), obstime=t)

        itrs_geo = teme.transform_to(ITRS(obstime=t))
        itrs_geo_p = itrs_geo.cartesian.without_differentials()
        itrs_geo_v = itrs_geo.cartesian.differentials['s']
        topo_itrs_p = itrs_geo_p - location.get_itrs(t).cartesian
        topo_itrs_repr = topo_itrs_p.with_differentials(itrs_geo_v)
        itrs_topo = ITRS(topo_itrs_repr, obstime=t, location=location)
        self.aa = itrs_topo.transform_to(AltAz(obstime=t, location=location))
        
        return self.aa



if __name__ == "__main__":
    import telescope
    t = telescope.Telescope(1.0, 6.5, telescope.UC_MT_JOHN_OBS)
    vis_sats = find_visible_satellites(t.location, sat_horizon=15)
    for k in vis_sats:
        print(k, k[0].altaz_from(t.location, k[1][0]))