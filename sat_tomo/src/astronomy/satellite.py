import datetime as dt
import os
import cProfile
import pstats
import joblib
from multiprocessing import Pool, Manager, Pipe
from copy import deepcopy
import pickle

from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation, ITRS, AltAz
import astropy.units as u
import astropy.time as aTime

from skyfield.api import load, EarthSatellite, utc
import sgp4
from sgp4 import exporter

from astronomy import astronomy, telescope

import random
import log as logging

log = logging.get_logger(__name__)
logging.setup_logger(log, level=logging.logging.DEBUG, to_console=True)

DEFAULT_CATALOG_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'

def load_sun_moon_emphemeris():
    load('de421.bsp')

def get_satellite_catalog(url=None, k=200):
    """ Retrieve catalog of skyfield satellite objects from url """
    if url == None:
        url = DEFAULT_CATALOG_URL

    now = dt.datetime.now()
    try:
        with open("./data/satellites/time.updated", 'rb') as f:
            t = pickle.load(f)
            log.info(f" > Satellite TLEs last updated at {t}")
        
        reload = (now - t) > dt.timedelta(days=14)
    except FileNotFoundError:
        reload = True
    
    if reload:
        log.info(f" > Downloading new satellite TLEs for {now}")

    satellites = load.tle_file(url, reload=reload)

    if reload:
        with open("./data/satellites/time.updated", 'wb') as f:
            pickle.dump(now, f)

    satellites = random.sample(satellites, k)
    for s in satellites:
        s.__class__ = Satellite
        s.picklable = False
    return satellites

def find_visible_satellites(location, satellites=None, time=None, tdelta=24, sat_horizon=0, sun_horizon=0, url=None, eph=None, number=100):
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
        satellites = get_satellite_catalog(url, k=number)
        log.info(f" > Loaded {len(satellites)} satellites <")

    # Load planet and sun ephemeris data
    if eph == None:
        eph = load('de421.bsp')

    event_names = ['Rise above horizon', 'Culminate', 'Set below horizon']

    pos = location.wgs84_latlon

    visible_satellites = []

    log.info(f" > Finding satellite visibility events <")

    for satellite in satellites:
        t, events = satellite.find_events(pos, t0, t1, altitude_degrees=sat_horizon)

        if len(events) > 0:
            log.debug(f"{satellite} : {len(events)} satellite visibility events")

            diff = satellite - pos

            vis_times = []
            t_rise = t[0]
            t_set = t[0]

            for ti, event in zip(t, events):
                s = location.get_sun_altitude(eph, ti)  # sun position at time

                if s < sun_horizon:   # sun under the horizon
                    if event == 0:   # rising
                        t_rise = ti
                    if event == 2:   # setting
                        t_set = ti
                        vis_times = vis_times + find_visible_times_between(t_rise, t_set, diff, eph)

            if len(vis_times) > 0:
                visible_satellites += [(satellite, vt) for vt in vis_times]
            
    return visible_satellites

def find_visibility(location, sat, time=None, tdelta=24, sat_horizon=0, sun_horizon=0, eph=None):
    """ Get a list of visible satellites, with times they are in sunlight.
    Checks for sun and moon position, so no satellite passes during the day are returns, nor are satellites that are transitting the moon

    Returns a list of (Satellite, time) tuples, where time is a list of tuples giving the UTC times the satellite appears then disappears
    Inputs:
    location    - Location object giving position of observer
    sat         - a satellite object
    time        - datetime object, giving the start time when looking for satellites begins
    tdelta      - time in hours from start time to search for visible satellites
    sat_horizon - Altitude angle where satellites above this angle are deemed potentially visible (0 is the actual horizon from location)
    sun_horizon - Altitude angle where the sun is below the horizon (0 is the actual horizon from location)
    eph         - planetary ephemeris data for determining sun, earth and moon locations
    """
    # Get time to find satellites
    ts = load.timescale()
    if time == None:
        time = dt.datetime.now(tz=dt.timezone.utc)
    t0 = ts.from_datetime(time)
    t1 = ts.from_datetime(time + dt.timedelta(hours=tdelta))

    # Load planet and sun ephemeris data
    if eph == None:
        eph = load('de421.bsp')

    event_names = ['Rise above horizon', 'Culminate', 'Set below horizon']

    pos = location.wgs84_latlon

    visibility = []

    # print(t0, t1)

    t, events = sat.find_events(pos, t0, t1, altitude_degrees=sat_horizon)

    if len(events) > 0:
        diff = sat - pos

        vis_times = []
        t_rise = t[0]
        t_set = t[0]

        for ti, event in zip(t, events):
            s = location.get_sun_altitude(eph, ti)  # sun position at time

            if s < sun_horizon:   # sun under the horizon
                if event == 0:   # rising
                    t_rise = ti
                if event == 2:   # setting
                    t_set = ti
                    vis_times = vis_times + find_visible_times_between(t_rise, t_set, diff, eph)

        if len(vis_times) > 0:
            visibility = [(sat, vt) for vt in vis_times]
            
    return visibility

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
    def get_tle(self):
        return exporter.export_tle(self.model)

class SatelliteMap():
    def __init__(self, tele=None, n_sats=500, star_magnitude=14, time_delta=30, time_period=72, asterism_size=3):
        self.n_processes = 8
        self.n_workers = 5

        self.n_sats=n_sats

        self.PROFILE_PROCESSES = True

        if tele == None:
            self.tele = telescope.default_telescope()
        else:
            self.tele = tele

        self.star_mags = list(range(6, star_magnitude + 1))

        self.time_period = time_period
        self.dt = dt.timedelta(seconds=time_delta)
        self.time = dt.datetime.now(tz=dt.timezone.utc)

        self.asterism_size=asterism_size

    def build(self):
        log.info(" > Building satellite map")

        manager = Manager()
        sats = manager.Queue()
        angles = manager.Queue()
        lock = manager.Lock()

        log.info(" > Setting up satellite queue <")
        satellites = get_satellite_catalog(k=self.n_sats)
        N = len(satellites)
        for i, sat in enumerate(satellites):
            sats.put((i, sat.get_tle()))
        log.info(f" > Finished setting up satellite queue - {N} satellites <")
        
        if sats.empty():
            log.info(" > NO SATELLITES IN QUEUE")
            return

        log.debug(" > Process satellites")

        n_sm_workers = self.n_processes - self.n_workers
        pipes = [Pipe() for _ in range(self.n_workers * n_sm_workers)]

        w_pipes = [p1 for p1, _ in pipes]
        w_pipes = [w_pipes[i:i + n_sm_workers] for i in range(0, len(pipes), n_sm_workers)]

        sm_pipes = [p2 for _, p2 in pipes]
        sm_pipes = [sm_pipes[x::n_sm_workers] for x in range(n_sm_workers)]
        
        mags = [self.star_mags[x::n_sm_workers] for x in range(n_sm_workers)]

        with Pool(processes=self.n_processes) as p:
            star_multi = [p.apply_async(self.starmap_worker, args=(m0, p2s, i)) for i, (m0, p2s) in enumerate(zip(mags, sm_pipes))]
            log.debug(f" > > Processes 'starmap_worker' started [{len(star_multi)}]")

            multi = [p.apply_async(self.satellite_worker, args=(N, p1, sats, angles, lock, i)) for i, p1 in enumerate(w_pipes)]
            log.debug(f" > > Processes 'worker' started [{len(multi)}]")

            res = [a.get() for a in multi]
            log.debug(" > > Processes 'worker' finished")

            for p1, p2 in pipes:
                p1.send("DONE")
            res = [a.get() for a in star_multi]
            log.debug(" > > Processes 'starmap_worker' finished")


        for p1, p2 in pipes:
            p1.close()
            p2.close()
            
        
        distances = {}
        for m in self.star_mags:
            distances[m] = []
        while not angles.empty():
            tmp = angles.get()
            distances[tmp[0]].append(tmp[1])

        log.info(" > Processing satellite orbits completed")

        log.info(" > Saving satellite data")
        
        for m, d in distances.items():    
            fn_pak = f"data/satellites/angle_data_mag{m}.pak"
            with open(fn_pak, 'rb') as f:
                log.debug(f"Loading old angle data from <{fn_pak}>")
                angles = joblib.load(f)

            angles = angles + d

            fn_temp = f"data/satellites/angle_data_mag{m}.temp"
            with open(fn_temp, 'wb') as f:
                log.debug(f"Saving new angle data to <{fn_temp}>")
                joblib.dump(angles, f)

            fn_old = f"data/satellites/angle_data_mag{m}.old"
            try:
                os.rename(fn_pak, fn_old)
                log.debug(f"Renaming <{fn_pak}> to <{fn_old}>")
            except FileExistsError:
                os.replace(fn_pak, fn_old)
                log.debug(f"Replacing <{fn_pak}> with <{fn_old}>")

            log.debug(f"Replacing <{fn_temp}> with <{fn_pak}>")
            os.replace(fn_temp, fn_pak)

        log.info(" > Completed building satellite map")
        return # build()

    def satellite_worker(self, N, conns, sats, angles, lock, wid):

        pid = os.getpid()
        log.info(f"> > [{pid:5}] PROCESS worker {wid} STARTING")

        if sats.empty():
            log.debug(f"> > [{pid:5}](W{wid}) No Satellite events present, ENDING PROCESS")
            return
        
        telescope = self.tele.copy()

        conn_dict = {}

        log.debug(f"> > [{pid:5}](W{wid}) Getting magnitude information")

        for conn in conns:
            conn.send("M")
            m = conn.recv()
            for mag in m:
                conn_dict[mag] = conn

        log.debug(f"> > [{pid:5}](W{wid}) Magnitude info received")

        count = 0

        lock.acquire()
        while not sats.empty():
            i, (tle1, tle2) = sats.get()
            lock.release()
            sat = Satellite(tle1, tle2)

            log.info(f"> > [{pid:5}](W{wid}) {(i+1):5}/{N}. Processing satellite: {sat}")
            events = find_visibility(telescope.location, sat=sat, time=self.time, tdelta=self.time_period)
            log.info(f"> > [{pid:5}](W{wid}) Found {len(events)} visibility events")

            for vis, (t0, t1) in events:

                log.debug(f"> > [{pid:5}](W{wid}) Processing event for {t0} to {t1}")

                telescope.set_target(vis, t0)
                times = []
                t = t0
                while t1 - t > 0:
                    times.append(t)
                    t += self.dt

                frames = len(times)
                log.debug(f"> > [{pid:5}](W{wid}) {frames} frames to compute")
                for j, _ in enumerate(times):
                    if (j + 1) % 256 == 0:
                        log.info(f"> > [{pid:5}](W{wid}) Frame {j+1}/{frames}")
                    count += 1
                    for m in self.star_mags:
                        conn_dict[m].send((m, telescope.orientation))
                        m_angle = conn_dict[m].recv()
                        angles.put(m_angle)
                        if count % 256 == 0:
                            log.debug(f"> > [{pid:5}](W{wid}) Angle data {m}:{m_angle}")
                    if count % 256 == 0:
                        count = 0

                    telescope.step(self.dt)
            
            lock.acquire()

        lock.release()

        # conn.close()
        log.debug(f"> > [{pid:5}] PROCESS worker {wid} ENDING")
        
        return # satellite_worker()
    
    def starmap_worker(self, mags, conns, wid):

        pid = os.getpid()
        log.info(f"* * [{pid:5}] PROCESS starmap_worker {wid} STARTING")

        log.info(f"* * [{pid:5}](S{wid}) Building starmaps {mags}")
        star_maps = {}
        stars = None
        for m in mags:
            log.debug(f"* * [{pid:5}](S{wid}) Star map for magnitude {m}")
            if os.path.exists(f"data/stars/starmap_{m}.pkl"):
                log.debug(f"* * [{pid:5}](S{wid}) Loading starmap: mag {m}")
                with open(f"data/stars/starmap_{m}.pkl", 'rb') as f: 
                    log.debug(f"* * [{pid:5}](S{wid}) Loading starmap from 'data/stars/starmap_{m}.pkl'")
                    star_maps[m] = joblib.load(f)
                    log.debug(f"* * [{pid:5}](S{wid}) Loaded starmap from 'data/stars/starmap_{m}.pkl'")
                log.debug(f"* * [{pid:5}](S{wid}) Loaded starmap: mag {m}")
            else:
                log.debug(f"* * [{pid:5}](S{wid}) Building starmap: mag {m}")
                if stars is None:
                    log.debug(f"* * [{pid:5}](S{wid}) Loading star data from catalog")
                    stars = astronomy.StarData()
                    log.debug(f"* * [{pid:5}](S{wid}) Star data loaded")
                star_maps[m] = astronomy.build_star_map(stars, mag=m, location=self.tele.location, time=self.time)
                with open(f'data/stars/starmap_{m}.pkl', 'wb') as f:
                    log.debug(f"* * [{pid:5}](S{wid}) Saving starmap to 'data/stars/starmap_{m}.pkl'")    
                    joblib.dump(star_maps[m], f)
                    log.debug(f"* * [{pid:5}](S{wid}) Saved starmap to 'data/stars/starmap_{m}.pkl'")
                log.debug(f"* * [{pid:5}](S{wid}) Completed starmap: mag {m}")
        log.info(f"* * [{pid:5}](S{wid}) Starmaps complete")
        stars = None # let garbage collection drop star data

        run = True
        count = 0

        while run:
            for conn in conns:
                if conn.poll():
                    msg = conn.recv()
                    if msg == "DONE":
                        run = False
                        break
                    elif msg == "M":
                        log.info(f"* * [{pid:5}]({wid}) Magnitude requested - sending {mags}")
                        conn.send(mags)
                    else:
                        count += 1
                        m, o = msg
                        try:
                            if m in mags:
                                if count % 256 == 0:
                                    log.info(f"* * [{pid:5}]({wid}) Responding to <{m}>:<{o}>")
                                angles = star_maps[m].closest(o, self.asterism_size, key=astronomy.get_angles_between)
                                if count % 256 == 0:
                                    log.debug(f"* * [{pid:5}]({wid}) <{m}>:<{o}> gives {angles}")

                                conn.send((m, angles))
                        except Exception as e:
                            log.info(f"* * [{pid:5}]({wid}) {e}")
                            quit()
        
        log.info(f"* * [{pid:5}] PROCESS starmap_worker {wid} ENDING")

        return # starmap_worker()
    
def load_map(mag):
    fn_pak = f"data/satellites/angle_data_mag{mag}.pak"
    with open(fn_pak, 'rb') as f:
        log.debug(f"Loading statellite angle data from <{fn_pak}>")
        angles = joblib.load(f)

    return angles

def clean_maps():
    for m in range(6, 14 + 1):
        fn_pak = f"data/satellites/angle_data_mag{m}.pak"
        fn_temp = f"data/satellites/angle_data_mag{m}.temp"
        fn_old = f"data/satellites/angle_data_mag{m}.old"

        with open(fn_old, 'rb') as f:
            log.debug(f"Loading old angle data from <{fn_old}>")
            angles = joblib.load(f)

        print(angles[0])
        print(angles[-1])

        new_angles = []
        for set in angles:
            try:
                new_set = []
                for a in set:
                    a1, a2, mag = a[0][0], a[0][1], a[1]
                    new_set.append((a1, a2, mag))
                new_angles.append(new_set)
            except TypeError:
                pass
            except IndexError:
                pass

        print(new_angles[0])
        print(new_angles[-1])

        with open(fn_temp, 'wb') as f:
            log.debug(f"Saving new angle data to <{fn_temp}>")
            joblib.dump(new_angles, f)

        try:
            os.rename(fn_temp, fn_pak)
            log.debug(f"Renaming <{fn_temp}> to <{fn_pak}>")
        except FileExistsError:
            os.replace(fn_temp, fn_pak)
            log.debug(f"Replacing <{fn_temp}> with <{fn_pak}>")

        # log.debug(f"Replacing <{fn_temp}> with <{fn_pak}>")
        # os.replace(fn_temp, fn_pak)

def build_statellite_map():
    load('de421.bsp')
    tele = telescope.default_telescope()
    sats = SatelliteMap(tele, time_period=24)
    sats.build()