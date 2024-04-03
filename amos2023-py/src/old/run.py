import datetime
from multiprocessing import Pool, Manager, Pipe
from copy import deepcopy
import joblib
import os
import cProfile
import pstats

from astro import telescope, astro, satellite

# settings
D = 1.8
f_number = 6.5
asterism_size = 3
star_magnitude = 14
time_period = 72
n_sats = 500

n_processes = 8
n_workers = 5

PROFILE_PROCESSES = True

location = telescope.UC_MT_JOHN_OBS
star_mags = list(range(6, star_magnitude + 1))

dt = datetime.timedelta(seconds=1)
time = datetime.datetime.now(tz=datetime.timezone.utc)
tele = telescope.Telescope(D, f_number, location)

def worker(N, conns, sats, dists, lock, wid):
    if PROFILE_PROCESSES:
        p = cProfile.Profile()
        p.enable()

    pid = os.getpid()
    print(f"> > [{pid:5}] PROCESS worker {wid} STARTING")
    
    if sats.empty():
        print(f"> > [{pid:5}](W{wid}) No Satellite events present, ENDING PROCESS")
        return 
    
    scope = telescope.Telescope(D, f_number, location)

    conn_dict = {}

    print(f"> > [{pid:5}](W{wid}) Getting magnitude information")

    for conn in conns:
        conn.send("M")
        m = conn.recv()
        for mag in m:
            conn_dict[mag] = conn

    print(f"> > [{pid:5}](W{wid}) Magnitude info received")
    
    lock.acquire()
    while not sats.empty():
        i, (tle1, tle2) = sats.get()
        lock.release()
        sat = satellite.Satellite(tle1, tle2)

        print(f"> > [{pid:5}](W{wid}) {(i+1):5}/{N}. Processing satellite: {sat}", flush=True)
        events = satellite.find_visible_satellites(tele.location, satellites=[sat], time=time, tdelta=time_period, reload=True, number=n_sats)
        print(f"> > [{pid:5}](W{wid}) Found {len(events)} visibility events", flush=True)

        for vis, (t0, t1) in events:

            print(f"> > [{pid:5}](W{wid}) Processing event for {t0} to {t1}", flush=True)

            scope.set_target(vis, t0)
            times = []
            t = t0
            while t1 - t > 0:
                times.append(t)
                t += dt

            print(f"> > [{pid:5}](W{wid}) {len(times)} frames to compute")
            for _ in times:
                for m in star_mags:
                    conn_dict[m].send((m, scope.orientation))
                    ds = conn_dict[m].recv()
                    
                    for n, d in enumerate(ds):
                        dists.put((m, n + 1, d * 3600)) # convert to arcseconds

                scope.step(dt)
        
        lock.acquire()

    lock.release()

    # conn.close()
    print(f"> > [{pid:5}] PROCESS worker {wid} ENDING")

    if PROFILE_PROCESSES:
        p.disable()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(p).strip_dirs().sort_stats(sortby)
        print(f"> > [{pid:5}] PROCESS worker {wid} PROFILE < < ")
        ps.print_stats(10)
        print(f"> > [{pid:5}] PROCESS worker {wid} PROFILE < < ")
    
    return # worker()

def starmap_worker(mags, conns, wid):
    if PROFILE_PROCESSES:
        p = cProfile.Profile()
        p.enable()

    pid = os.getpid()
    print(f"* * [{pid:5}] PROCESS starmap_worker {wid} STARTING")

    print(f"* * [{pid:5}](S{wid}) Building starmaps {mags}", flush=True)
    star_maps = {}
    stars = None
    for m in mags:
        if os.path.exists(f"data/starmaps/starmap_{m}.pkl"):
            print(f"* * [{pid:5}](S{wid}) Loading starmap: mag {m}")
            with open(f"data/starmaps/starmap_{m}.pkl", 'rb') as f: 
                star_maps[m] = joblib.load(f)
            print(f"* * [{pid:5}](S{wid}) Loaded starmap: mag {m}", flush=True)
        else:
            print(f"* * [{pid:5}](S{wid}) Building starmap: mag {m}")
            if stars is None:
                print(f"* * [{pid:5}](S{wid}) Loading star data from catalog")
                stars = astro.StarData()
                print(f"* * [{pid:5}](S{wid}) Star data loaded")
            star_maps[m] = astro.StarMap(stars, mag=m, location=tele.location, time=time)
            with open(f'data/starmaps/starmap_{m}.pkl', 'wb') as f:
                joblib.dump(star_maps[m], f)
            print(f"* * [{pid:5}](S{wid}) Completed starmap: mag {m}", flush=True)
    print(f"* * [{pid:5}](S{wid}) Starmaps complete", flush=True)
    stars = None # let garbage collection drop star data

    run = True

    while run:
        for conn in conns:
            if conn.poll():
                msg = conn.recv()
                if msg == "DONE":
                    run = False
                    break
                elif msg == "M":
                    print(f"* * [{pid:5}]({wid}) Magnitude requested")
                    conn.send(mags)
                else:
                    m, o = msg
                    if m in mags:
                        _, ds = star_maps[m].closest(o, asterism_size)
                        conn.send(ds)
    
    print(f"* * [{pid:5}] PROCESS starmap_worker {wid} ENDING", flush=True)
    if PROFILE_PROCESSES:
        p.disable()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(p).strip_dirs().sort_stats(sortby)
        print(f"* * [{pid:5}] PROCESS starmap_worker {wid} PROFILE * *", flush=True)
        ps.print_stats(10)
        print(f"* * [{pid:5}] PROCESS starmap_worker {wid} PROFILE * *", flush=True)

    return # starmap_worker()

def main():
    manager = Manager()
    sats = manager.Queue()
    dists = manager.Queue()
    lock = manager.Lock()

    print("> Set up satellite queue")
    satellites = satellite.get_satellite_catalog(reload=True, k=n_sats)
    N = len(satellites)
    for i, sat in enumerate(satellites):
        sats.put((i, sat.get_tle()))
    print(f"> Finished setting up satellite queue - {N} satellites")
    
    if sats.empty():
        print("> NO SATELLITES")
        return

    print("> Process satellites")

    n_sm_workers = n_processes - n_workers
    pipes = [Pipe() for _ in range(n_workers * n_sm_workers)]

    w_pipes = [p1 for p1, _ in pipes]
    w_pipes = [w_pipes[i:i + n_sm_workers] for i in range(0, len(pipes), n_sm_workers)]

    sm_pipes = [p2 for _, p2 in pipes]
    sm_pipes = [sm_pipes[x::n_sm_workers] for x in range(n_sm_workers)]
    
    mags = [star_mags[x::n_sm_workers] for x in range(n_sm_workers)]

    with Pool(processes=n_processes) as p:
        star_multi = [p.apply_async(starmap_worker, args=(m0, p2s, i)) for i, (m0, p2s) in enumerate(zip(mags, sm_pipes))]
        print(f"> > Processes 'starmap_worker' started [{len(star_multi)}]")

        multi = [p.apply_async(worker, args=(N, p1, sats, dists, lock, i)) for i, p1 in enumerate(w_pipes)]
        print(f"> > Processes 'worker' started [{len(multi)}]")

        processes = star_multi + multi

        # while not all([a.ready() for a in multi]):
        #     for p in processes:
        #         if p.ready():
        #             p.get()

        res = [a.get() for a in multi]
        print("> > Processes 'worker' finished")

        for p1, p2 in pipes:
            p1.send("DONE")
        res = [a.get() for a in star_multi]
        print("> > Processes 'starmap_worker' finished")


    for p1, p2 in pipes:
        p1.close()
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
        with open(f"data/satellites/data_{m}_{n}.pak", 'wb') as f:
            joblib.dump(d, f)

    return # main()

if __name__ == "__main__":
    p = cProfile.Profile()

    print("<> <> <> START <> <> <>")
    p.enable()
    main()
    p.disable()

    print("<> <> <>  END  <> <> <>")

    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(p).strip_dirs().sort_stats(sortby)
    ps.print_stats(10)