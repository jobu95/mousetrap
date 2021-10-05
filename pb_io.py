import numpy as np
import os
import frames_pb2

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
        theta = np.arctan(zero_y * 1.0 / zero_x)
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
        new_pos.sx = int(new_x * np.cos(theta) - new_y * np.sin(theta))
        new_pos.sy = int(new_x * np.sin(theta) + new_y * np.cos(theta))
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

# Convert an arc in absolute positioning format to delta positioning format.
# The only data that's lost is the arc's original position in the frame.
def arc_abs_to_delta(arc):
    new_arc = frames_pb2.MousePositions()
    prev = None
    for pos in arc.positions:
        if prev == None:
            prev = pos
            continue

        #print("x: {},{}".format(prev.x, pos.x))
        #print("y: {},{}".format(prev.y, pos.y))
        new_pos = new_arc.positions.add()
        new_pos.sx = pos.x - prev.x
        new_pos.sy = pos.y - prev.y
        #print("dx: {}".format(new_pos.sx))
        #print("dy: {}".format(new_pos.sy))

        prev = pos

    return new_arc

def arcs_abs_to_delta(arcs):
    new_arcs = frames_pb2.Arcs()
    for arc in arcs.arcs:
        new_arcs.arcs.append(arc_abs_to_delta(arc))
    return new_arcs

def arc_delta_to_abs(arc):
    new_arc = frames_pb2.MousePositions()
    prev = new_arc.positions.add()
    prev.sx = 0
    prev.sy = 0
    prev.time = 0
    for pos in arc.positions:
        new_pos = new_arc.positions.add()
        new_pos.sx = pos.sx + prev.sx
        new_pos.sy = pos.sy + prev.sy

        prev = new_pos

    return new_arc

def arcs_delta_to_abs(arcs):
    new_arcs = frames_pb2.Arcs()
    for arc in arcs.arcs:
        new_arcs.arcs.append(arc_delta_to_abs(arc))
    return new_arcs

# Generate a map of arcs where the key = log2(len(arc)).
def bucketize_arcs_len(arcs):
    arcs_of_log2_length = {}

    for arc in arcs.arcs:
        start = arc.positions[0]
        end = arc.positions[-1]
        dx = end.x - start.x
        dy = end.y - start.y
        dist = int(np.sqrt(dx * dx + dy * dy))
        dist_log2 = 0
        if dist >= 2:
            dist_log2 = int(np.ceil(np.log2(dist)))
        if dist_log2 not in arcs_of_log2_length:
            arcs_of_log2_length[dist_log2] = frames_pb2.Arcs()
        arcs_of_log2_length[dist_log2].arcs.append(arc)

    return arcs_of_log2_length

# Generate a map of arcs where the key = log2(duration(arc)).
def bucketize_arcs_dur(arcs):
    arcs_of_log2_length = {}

    for arc in arcs.arcs:
        dt = len(arc.positions)
        dist_log2 = int(np.ceil(np.log2(dt)))
        if dist_log2 not in arcs_of_log2_length:
            arcs_of_log2_length[dist_log2] = frames_pb2.Arcs()
        arcs_of_log2_length[dist_log2].arcs.append(arc)

    return arcs_of_log2_length

def delta_bounding_box(darcs):
    min_dx = 1000
    max_dx = 0
    min_dy = 1000
    max_dy = 0
    for arc in darcs.arcs:
        for pos in arc.positions:
            if pos.sx < min_dx:
                min_dx = pos.sx
            if pos.sx > max_dx:
                max_dx = pos.sx
            if pos.sy < min_dy:
                min_dy = pos.sy
            if pos.sy > max_dy:
                max_dy = pos.sy
    print("min dx: {}".format(min_dx))
    print("max dx: {}".format(max_dx))
    print("min dy: {}".format(min_dy))
    print("max dy: {}".format(max_dy))
    return min_dx, max_dx, min_dy, max_dy

# Map a set of delta arcs to a numpy array.
# 1 - calculate longest duration of an arc.
# 2 - calculate bounds on x and y deltas (min/max of each)
# 3 - map deltas to [0, 255].
# 2 - allocate a (duration, 2) np array of uint8_t's.
# 3 - populate the array.
def darcs_to_array(darcs):
    num_arcs = len(darcs.arcs)
    max_dur = 0
    (min_dx, max_dx, min_dy, max_dy) = delta_bounding_box(darcs)
    for arc in darcs.arcs:
        if len(arc.positions) > max_dur:
            max_dur = len(arc.positions)
    # Round up to nearest power of 2
    max_dur += 1
    print("max_dur: {}".format(max_dur))

    # num_arcs rows
    # max_dur columns
    # 2 copies (one for x, one for y)
    data = np.zeros((num_arcs, max_dur, 2), dtype=np.float32)

    # using dcgan as a reference, we want our data normalized on [-1, 1].
    n_arc = 0
    for arc in darcs.arcs:
        n_pos = 0
        for pos in arc.positions:
            #print("pos: {},{}".format(pos.sx, pos.sy))
            # scale to [-1.0, 1.0]
            dx = pos.sx * 1.0 / max(np.abs(max_dx), np.abs(min_dx), 1)
            dy = pos.sy * 1.0 / max(np.abs(max_dy), np.abs(min_dy))
            # store in data
            data[n_arc, n_pos, 0] = dx
            data[n_arc, n_pos, 1] = dy

            n_pos += 1
        n_arc += 1

    return data

def array_to_darcs(raw_arcs, min_dx, max_dx, min_dy, max_dy, framerate):
    arcs = frames_pb2.Arcs()
    i = 0
    dbg_every = 32000
    for raw_arc in raw_arcs:
        i += 1
        if i % dbg_every == 0:
            print("ARC")
        arc = arcs.arcs.add()
        time = 0.0
        for raw_pos in raw_arc:
            if i % dbg_every == 0:
                print("raw pos: {},{}".format(raw_pos[0], raw_pos[1]))
            pos = arc.positions.add()

            # Map from [-1, 1] to original domain
            pos.sx = int(raw_pos[0] * max(np.abs(max_dx), np.abs(min_dx), 1))
            pos.sy = int(raw_pos[1] * max(np.abs(max_dy), np.abs(min_dy), 1))
            # Infer relative timestamp.
            pos.time = time
            time += 1.0 / framerate
    return arcs
