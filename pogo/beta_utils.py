def ref_h(floor, ceil):
    return min(max(floor + 1.4, 0.6 * ceil + 0.4 * floor), ceil)