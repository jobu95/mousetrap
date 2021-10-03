import os
import pyautogui as gui
import threading
import time
import frames_pb2

def replay_arc(pb_mouse_positions):
    prev_mouse_s = 0
    now_s = time.monotonic()
    start_s = now_s
    gui.PAUSE = 0
    i = 0
    for mouse_pos in pb_mouse_positions.positions:
        # Sleep until the next frame time.
        mouse_delay = (mouse_pos.time - prev_mouse_s)
        real_delay = time.monotonic() - now_s
        while prev_mouse_s > 0 and real_delay < mouse_delay:
            time.sleep(0.01)
            real_delay = time.monotonic() - now_s
        if prev_mouse_s > 0:
            now_s += mouse_delay
        prev_mouse_s = mouse_pos.time

        # Debug information
        i += 1
        #print("{}: ({}, {})".format(mouse_pos.time, mouse_pos.x, mouse_pos.y))

        # Move mouse
        gui.moveTo(mouse_pos.x, mouse_pos.y)
    print("Replayed {} mouse positions over {} seconds".format(i, time.monotonic() - start_s))
    gui.PAUSE = 1

def replay_arcs(pb_arcs, delay_s = 0):
    for arc in pb_arcs.arcs:
        if delay_s > 0:
            time.sleep(delay_s)
        print("Playing arc")
        replay_arc(arc)
