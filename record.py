import os
import pyautogui as gui
import threading
import time
import frames_pb2

import pb_io

class ThreadArgs:
    def __init__(self):
        pass

    run = True

    lock = threading.Lock()

    # list of positions.
    # each position is represented as ((x, y), time).
    # protected by lock.
    mouse_positions = []

    # list of dropped frames.
    # each is a simple monotonic timestamp.
    # used for debugging.
    # protected by lock.
    dropped_frames = []

def watch(args):
    # core framerate management
    goal_framerate = 60 # in frames per second
    goal_loop_time_s = 1.0 / goal_framerate
    start_s = time.monotonic()
    next_s = start_s + goal_loop_time_s

    # logging
    log_cadence_s = 5.0 # log every n seconds
    log_s = start_s + log_cadence_s

    # framerate monitoring
    i = 0
    prev = time.monotonic()

    print("goal loop time: {}".format(goal_loop_time_s))
    while args.run:
        # sleep until next frame window
        now_s = time.monotonic()
        while now_s < next_s:
            time.sleep(0.01)
            now_s = time.monotonic()
        now_s = time.monotonic()
        next_s += goal_loop_time_s

        # if next frame is behind present, then drop frames till we hit
        # present. This avoids pinning the CPU if we hit a lag spike.
        while next_s < now_s:
            args.lock.acquire()
            args.dropped_frames.append(next_s)
            args.lock.release()
            next_s += goal_loop_time_s

        # record mouse position
        args.lock.acquire()
        args.mouse_positions.append((gui.position(), now_s))
        args.lock.release()

        # log every (goal framerate) frames
        i += 1
        if i % goal_framerate == 0:
            print("{} frames in {} seconds".format(goal_framerate, now_s - prev))
            prev = now_s

        # log every log_s seconds
        if now_s > log_s:
            print("watch_thread stats")
            args.lock.acquire()
            print(" mouse positions seen: {}".format(len(args.mouse_positions)))
            print(" dropped frames: {}".format(len(args.dropped_frames)))
            args.lock.release()
            log_s += log_cadence_s

def watch_thread(args):
    print("watch_thread started")
    watch(args)
    print("watch_thread stopped")

def log(args):
    goal_loop_time_s = 30.0
    next_s = time.monotonic() + goal_loop_time_s

    if not os.path.isdir("data"):
        os.mkdir("data")
    mouse_pos_dir = "data/mouse_pos"
    if not os.path.isdir(mouse_pos_dir):
        os.mkdir(mouse_pos_dir)
    dropped_frames_dir = "data/dropped_frames"
    if not os.path.isdir(dropped_frames_dir):
        os.mkdir(dropped_frames_dir)

    while args.run:
        # sleep until next frame window
        now_s = time.monotonic()
        while args.run and now_s < next_s:
            now_s = time.monotonic()
            time.sleep(1)
        next_s += goal_loop_time_s

        # grab data and release locks
        print("log_thread grabbing data...")
        mouse_positions = []
        dropped_frames = []
        args.lock.acquire()
        mouse_positions = args.mouse_positions
        args.mouse_positions = []
        dropped_frames = args.dropped_frames
        args.dropped_frames = []
        args.lock.release()
        print("log_thread grabbed data")

        # migrate data to proto
        pb_mouse_positions = frames_pb2.MousePositions()
        for mouse_pos in mouse_positions:
            pb_mouse_pos = pb_mouse_positions.positions.add()
            pb_mouse_pos.time = mouse_pos[1]
            pb_mouse_pos.x = mouse_pos[0][0]
            pb_mouse_pos.y = mouse_pos[0][1]
        pb_dropped_frames = []
        for dropped_frame in dropped_frames:
            pb_dropped_frame = frames_pb2.DroppedFrame()
            pb_dropped_frame.time = dropped_frame
            pb_dropped_frames.append(pb_dropped_frame)

        # save mouse positions to disk
        now = time.time()
        filename = "{}/{}.pb".format(mouse_pos_dir,now)
        print("save {} mouse pos to {}".format(len(pb_mouse_positions.positions), filename))
        pb_io.save_pb2(pb_mouse_positions, filename)

        # TODO save dropped frames

def log_thread(args):
    print("log_thread started")
    log(args)
    print("log_thread stopped")

def record():
    thread_args = ThreadArgs()
    watch_thd = threading.Thread(target=watch_thread, args=(thread_args,))
    log_thd = threading.Thread(target=log_thread, args=(thread_args,))

    print("Main thread starting watch_thread")
    watch_thd.start()
    print("Main thread starting log_thread")
    log_thd.start()

    print("Press enter to exit the application")
    input()
    thread_args.run = False
    print("Main thread joining watch_thread")
    watch_thd.join()
    print("Main thread joining log_thread")
    log_thd.join()
