import os
import pyautogui as gui
import threading
import time
import frames_pb2

from numpy import arctan as np_arctan
from numpy import cos as np_cos
from numpy import sin as np_sin

def save_pb2(pb2, path):
    f = open(path, "wb")
    f.write(pb2.SerializeToString())
    f.close()

def load_pb2(pb, pb_path):
    f = open(pb_path, "rb")
    pb.ParseFromString(f.read())
    f.close()
    return pb

def extract_arcs(pb_mouse_positions):
    # We will find arcs by advancing two pointers through the list.
    arc_begin = None
    arc_end = None
    arcs = frames_pb2.Arcs()
    arc = frames_pb2.MousePositions()
    for mouse_pos in pb_mouse_positions.positions:
        # Advance the 'begin' pointer until there's movement
        if arc_begin == None:
            arc_begin = mouse_pos
            continue
        if mouse_pos.x == arc_begin.x and mouse_pos.y == arc_begin.y:
            arc_begin = mouse_pos
            continue
        # Advance the 'end' pointer until there's no more movement
        arc.positions.append(mouse_pos)
        if arc_end == None:
            arc_end = mouse_pos
            continue
        if mouse_pos.x != arc_end.x or mouse_pos.y != arc_end.y:
            arc_end = mouse_pos
            continue

        if len(arc.positions) >= 5:
            arcs.arcs.append(arc)

        arc = frames_pb2.MousePositions()
        arc_begin = None
        arc_end = None
    return arcs

def extract_all_arcs(directory):
    all_arcs = frames_pb2.Arcs()
    for filename in os.listdir(directory):
        if not filename.endswith(".pb"):
            continue
        arcs = extract_arcs(load_pb2(frames_pb2.MousePositions(), os.path.join(directory, filename)))
        all_arcs.arcs.extend(arcs.arcs)
    print("Total # of arcs: {}".format(len(all_arcs.arcs)))
    return all_arcs

def center_arc(arc):
    pi = 3.14159265

    # Calculate angle from arc's first to last point.
    first_point = arc.positions[0]
    last_point = arc.positions[-1]
    zero_x, zero_y = (last_point.x - first_point.x,
            last_point.y - first_point.y)
    # 45* -> x = n, y = n -> input is 1 -> output is pi/4
    # 135* -> x = -n, y = n -> input is -1 -> output is -pi/4 -> add pi
    # 225* -> x = -n, y = -n -> input is 1 -> output is pi/4 -> pi
    # 315* -> x = n, y = -n -> input is -1 -> output is -pi/4 -> ADD 2 * pi
    if zero_x == 0:
        if zero_y > 0:
            theta = pi / 2
        else:
            theta = 3 * pi / 2
    else:
        theta = np_arctan(zero_y * 1.0 / zero_x)
        if zero_x < 0:
            theta += pi
        elif zero_y < 0:
            theta += 2 * 3.14159265358979
    theta *= -1
    theta += pi / 2

    new_arc = frames_pb2.MousePositions()
    for pos in arc.positions:
        new_pos = new_arc.positions.add()

        # Translate ray to (0,0)
        new_x = pos.x - first_point.x
        new_y = pos.y - first_point.y
        # Rotate so that it falls along the x-axis
        new_pos.sx = int(new_x * np_cos(theta) - new_y * np_sin(theta))
        new_pos.sy = int(new_x * np_sin(theta) + new_y * np_cos(theta))
        new_pos.time = pos.time
    return new_arc

def center_arcs(arcs):
    new_arcs = frames_pb2.Arcs()
    for arc in arcs.arcs:
        new_arcs.arcs.append(center_arc(arc))
    return new_arcs

# Convert signed arcs to unsigned by translating them into first quadrant.
def unsign_arcs(arcs):
    min_x = 1024 * 1024 * 1024 * 2
    min_y = 1024 * 1024 * 1024 * 2
    # Find bottom left corner of bounding box.
    for arc in arcs.arcs:
        for pos in arc.positions:
            if pos.sx < min_x:
                min_x = pos.sx
            if pos.sy < min_y:
                min_y = pos.sy
    # Translate signed arc positions to first quadrant of 2D plane.
    new_arcs = frames_pb2.Arcs()
    for arc in arcs.arcs:
        new_arc = new_arcs.arcs.add()
        for pos in arc.positions:
            new_pos = new_arc.positions.add()
            new_pos.x = pos.sx - min_x
            new_pos.y = pos.sy - min_y
            new_pos.time = pos.time
    return new_arcs

