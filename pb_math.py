import os
import pyautogui as gui
import threading
import time
import frames_pb2

def pb_distance2(pb_mouse_a, pb_mouse_b):
    dx = pb_mouse_a.x - pb_mouse_b.x
    dy = pb_mouse_a.y - pb_mouse_b.y
    return dx * dx + dy * dy
