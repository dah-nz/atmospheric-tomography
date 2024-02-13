import numpy as np
import datetime
from astropy.coordinates import AltAz

from multiprocessing import Pool, Manager, Pipe
from copy import deepcopy
import joblib
import os

from astro import telescope, astro, atmosphere, satellite
from tomography import tomo_LSQ, tomo_MLNN
from tomography.beacon import Beacon
from basis.DCT import DCT

# settings
D = 1.8
f_number = 6.5
asterism_size = 3
star_magnitude = 6
time_period = 72
n_sats = 10

location = telescope.UC_MT_JOHN_OBS
star_mags = list(range(6, star_magnitude + 1))

dt = datetime.timedelta(seconds=1)
time = datetime.datetime.now(tz=datetime.timezone.utc)
tele = telescope.Telescope(D, f_number, location)

def worker(N, conn, sats, dists, lock):
    print(f"> > [{os.getpid():5}] PROCESS STARTING")    
    
    if sats.empty():
        print(f"> > [{os.getpid()}:5] No Satellite events present, ENDING PROCESS")
        return 
    
    scope = telescope.Telescope(D, f_number, location)
    
    lock.acquire()
    while not sats.empty():
        i, (tle1, tle2) = sats.get()
        lock.release()
        sat = satellite.Satellite(tle1, tle2)

        print(f"> > [{os.getpid():5}] {(i+1):5}/{N}. Processing satellite: {sat}", flush=True)
        events = satellite.find_visible_satellites(tele.location, satellites=[sat], time=time, tdelta=time_period, reload=True, number=n_sats)
        print(f"> > [{os.getpid():5}] Found {len(events)} visibility events", flush=True)

        for vis, (t0, t1) in events:

            scope.set_target(vis, t0)
            times = []
            t = t0
            while t1 - t > 0:
                times.append(t)
                t += dt
            for _ in times:

                for m in star_mags:
                    # print("> > Sending starmap query", flush=True)
                    conn.send((m, scope.orientation))
                    ds = conn.recv()
                    # _, ds = star_maps[m].closest(scope.orientation, asterism_size)
                    
                    for n, d in enumerate(ds):
                        dists.put((m, n + 1, d * 3600)) # convert to arcseconds

                scope.step(dt)
        
        lock.acquire()

    lock.release()

    conn.close()
    print(f"> > [{os.getpid():5}] PROCESS ENDING")
    
    return # worker()

def main():
    manager = Manager()
    sats = manager.Queue()
    dists = manager.Queue()
    lock = manager.Lock()

    stars = None

    print("> Building starmaps")
    star_maps = {}
    for m in star_mags:
        if os.path.exists(f"data/starmaps/starmap_{m}.pkl"):
            print("> > Loading starmap: mag {}".format(m))
            with open(f"data/starmaps/starmap_{m}.pkl", 'rb') as f: 
                star_maps[m] = joblib.load(f)
            print("> > Loaded starmap: mag {}".format(m))
        else:
            print("> > Building starmap: mag {}".format(m))
            if stars is None:
                print("> Loading star data from catalog")
                stars = astro.StarData()
                print("> Star data loaded")
            star_maps[m] = astro.StarMap(stars, mag=m, location=tele.location, time=time)
            with open(f'data/starmaps/starmap_{m}.pkl', 'wb') as f:
                joblib.dump(star_maps[m], f)
            print("> > Completed starmap: mag {}".format(m))
    print("> Starmaps complete")

    print("> Set up satellite queue")
    satellites = satellite.get_satellite_catalog(k=n_sats)
    N = len(satellites)
    for i, sat in enumerate(satellites):
        sats.put((i, sat.get_tle()))
    print(f"> Finished setting up satellite queue - {N} satellites")
    
    if sats.empty():
        print("> NO SATELLITES")
        return

    print("> Process satellites")

    n_processes = 6
    pipes = [Pipe() for _ in range(n_processes)]

    with Pool(processes=n_processes) as p:
        multi = [p.apply_async(worker, args=(N, p1, sats, dists, lock)) for p1, _ in pipes]
        print("> > Processes started")
        while not all([a.ready() for a in multi]):
            for _, p2 in pipes:
                if p2.poll():
                    # print("> > Receiving starmap query")
                    m, o = p2.recv()
                    _, ds = star_maps[m].closest(o, asterism_size)
                    p2.send(ds)
        res = [a.get() for a in multi]
        print("> > Processes finished")

    for p1, p2 in pipes:
        p2.close()
    
    distances = {}
    for m in star_mags:
        for n in range(asterism_size):
            distances[(m, n + 1)] = []
    while not dists.empty():
        m, n, d = dists.get()
        distances[(m, n)].append(d)

    print("> Processing completed")

    print("> Saving data")
    for (m, n), d in distances.items():
        with open(f"data_{m}_{n}.pak", 'wb') as f:
            joblib.dump(d, f)

    return # main()

if __name__ == "__main__":
    print("<> <> <> START <> <> <>")
    main()
    print("<> <> <>  END  <> <> <>")