def chamfer(A, B):
    d1 = mean(min_distance(A_i, B))
    d2 = mean(min_distance(B_j, A))
    return d1 + d2

def hausdorff_distance(A, B, percentile=95):
    d1, _ = nearest_distances(A, B)
    d2, _ = nearest_distances(B, A)
    return max(
        np.percentile(d1, percentile),
        np.percentile(d2, percentile)
    )

def dtw_smoothness(path):
    slopes = []
    for i in range(1, len(path)):
        dt1 = path[i][0] - path[i-1][0]
        dt2 = path[i][1] - path[i-1][1]
        slopes.append(dt2 / (dt1 + 1e-6))

    return sum(abs(slopes[i] - slopes[i-1]) for i in range(1, len(slopes)))