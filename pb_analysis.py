import pb_io
import frames_pb2

def analyze_arcs(arcs):
    arcs_by_len = pb_io.bucketize_arcs_len(arcs)
    arcs_by_dur = pb_io.bucketize_arcs_dur(arcs)

    print("Total # of arcs: {}".format(len(arcs.arcs)))

    print("Histogram: # arcs of a given duration (frames)")
    total = 0
    for dur in sorted(arcs_by_dur):
        print("  {}: {}".format(dur, len(arcs_by_dur[dur].arcs)))
        total += len(arcs_by_dur[dur].arcs)
        print("  {}".format(total * 100.0 / len(arcs.arcs)))
    print("Histogram: # arcs of a given log2(length) (pixels)")
    total = 0
    for length in sorted(arcs_by_len):
        print("  {}: {}".format(length, len(arcs_by_len[length].arcs)))
        total += len(arcs_by_len[length].arcs)
        print("  {}".format(total * 100.0 / len(arcs.arcs)))
        print("  {}".format(len(arcs_by_len[length].arcs) * 100.0 / len(arcs.arcs)))

def get_interesting_arcs(arcs):
    # Based on analyze_arcs(), I saw that 75% of my mouse movements were under
    # 16 frames in duration.
    #
    # Orthogonally, my arcs were more uniformly distributed in terms of
    # distance. I chose an arbitrary but fairly well represented set of
    # distances to start with.
    print("Input len: {}".format(len(arcs.arcs)))
    buckets = pb_io.bucketize_arcs_dur(arcs)
    arcs2 = frames_pb2.Arcs()
    for i in range(0, 4 + 1):
        if not i in buckets:
            continue
        arcs2.arcs.extend(buckets[i].arcs)
    print("Arcs of duration 0-16: {}".format(len(arcs2.arcs)))
    
    buckets = pb_io.bucketize_arcs_len(arcs2)
    arcs2 = frames_pb2.Arcs()
    # Each of these buckets is ~10% of the data set.
    # Arcs of length [32, 256)
    for i in range(5, 7 + 1):
        if not i in buckets:
            continue
        arcs2.arcs.extend(buckets[i].arcs)
    print("Arcs of length 32-255: {}".format(len(arcs2.arcs)))
    return arcs2
