class Frame:
    def __init__(self, points, labels=None):
        self.points = points      # (N, 3)
        self.labels = labels      # (N,) or None


class Sequence:
    def __init__(self, frames):
        self.frames = frames      # list of Frame

class AlignmentResult:
    def __init__(self):
        self.aligned_B = []        # list of aligned B_t → A_t
        self.dtw_path = []         # [(t_A, t_B), ...]
        self.correspondences = []  # list per frame: [(i, j, prob)]