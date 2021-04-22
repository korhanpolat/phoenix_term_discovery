import numpy as np
from numba import jit, prange


def sdtw(dist_mat, w=2, start=[0, 0]):
    # adapted from..  https://github.com/talcs/simpledtw

    i0, j0 = start
    m, n = dist_mat.shape

    matrix = np.full((m, n), np.inf)
    matrix[max(0, i0 - w), max(0, j0 - w)] = dist_mat[max(0, i0 - w), max(0, j0 - w)]

    # initialize costs
    for i in range(max(0, i0 - w) + 1, min(m, i0 + w + 1)):
        matrix[i, 0] = dist_mat[i, 0] + matrix[i - 1, 0]

    for j in range(max(0, j0 - w) + 1, min(n, j0 + w + 1 - i0)):
        matrix[0, j] = dist_mat[0, j] + matrix[0, j - 1]

    # aggregate costs - dynamic programming
    for i in range(max(1, i0), m):
        for j in range(max(1, j0 + i - w - i0), min(n, j0 + i + w + 1 - i0)):
            cost = min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
            matrix[i, j] = dist_mat[i, j] + cost

    # find starting point of back-track
    loc = np.where(matrix == min(matrix[-1, :].min(), matrix[:, -1].min()))
    i, j = loc[0][-1], loc[1][-1]

    path = []

    while i > i0 or j > j0:
        path.append((i, j))

        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf

        move = np.argmin([option_diag, option_up, option_left])
        if move == 2:
            j -= 1
        elif move == 1:
            i -= 1
        else:
            i -= 1
            j -= 1

    #    path.append((i0,j0))
    path.reverse()

    return path, matrix[-1, -1], matrix


def LCMA(A, L, extend_r=0.0):
    assert L < len(A)

    cumsum = np.cumsum(A)
    i0 = 0
    j0 = len(cumsum) - 1
    min_mean = (cumsum[j0] - cumsum[i0]) / (j0 - i0)

    for i in range(len(cumsum)):
        for j in range(i + L, len(cumsum)):
            tmp_mean = (cumsum[j] - cumsum[i]) / (j - i)
            if tmp_mean < min_mean:
                min_mean = tmp_mean
                i0 = i
                j0 = j

    if extend_r > 0:
        extension = (j0 - i0) * extend_r
        i0 = max(0, int(i0 - extension/2) )
        j0 = min(len(A)-1, int(j0 + extension/2) )
        min_mean = (cumsum[j0] - cumsum[i0]) / (j0 - i0)
    return i0, j0, min_mean


@jit
def sdtw_np(dist_mat, w=2, start=[0, 0]):
    # adapted from..  https://github.com/talcs/simpledtw

    i0, j0 = start
    m, n = dist_mat.shape

    matrix = np.full((m, n), np.inf)
    matrix[max(0, i0 - w), max(0, j0 - w)] = dist_mat[max(0, i0 - w), max(0, j0 - w)]

    # initialize costs
    for i in range(max(0, i0 - w) + 1, min(m, i0 + w + 1)):
        matrix[i, 0] = dist_mat[i, 0] + matrix[i - 1, 0]

    for j in range(max(0, j0 - w) + 1, min(n, j0 + w + 1 - i0)):
        matrix[0, j] = dist_mat[0, j] + matrix[0, j - 1]

    # aggregate costs - dynamic programming
    for i in range(max(1, i0), m):
        for j in range(max(1, j0 + i - w - i0), min(n, j0 + i + w + 1 - i0)):
            cost = min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
            matrix[i, j] = dist_mat[i, j] + cost

    # find starting point of back-track
    loc = np.where(matrix == min(matrix[-1, :].min(), matrix[:, -1].min()))
    i, j = loc[0][-1], loc[1][-1]

    path = np.zeros((m + n, 2))
    cnt = -1

    while i > i0 or j > j0:
        path[cnt] = [i, j]
        cnt -= 1

        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf

        move = np.argmin((option_diag, option_up, option_left))
        if move == 2:
            j -= 1
        elif move == 1:
            i -= 1
        else:
            i -= 1
            j -= 1

#    path = path[cnt + 1:]

    return path, matrix[-1, -1], matrix


@jit()
def sdtw_jit(dist_mat, w=2, start=[0, 0], backtr_loc=None):
    # adapted from..  https://github.com/talcs/simpledtw
    # dist_mat -> m x n non negative matrix
    # w -> width of diagonal search path

    i0, j0 = start
    m, n = dist_mat.shape

    matrix = np.full((m, n), np.inf)
    matrix[max(0, i0 - w), max(0, j0 - w)] = dist_mat[max(0, i0 - w), max(0, j0 - w)]

    # initialize costs
    for i in range(max(0, i0 - w) + 1, min(m, i0 + w + 1)):
        matrix[i, 0] = dist_mat[i, 0] + matrix[i - 1, 0]

    for j in range(max(0, j0 - w) + 1, min(n, j0 + w + 1 - i0)):
        matrix[0, j] = dist_mat[0, j] + matrix[0, j - 1]

    # aggregate costs - dynamic programming
    for i in range(max(1, i0), m):
        for j in range(max(1, j0 + i - w - i0), min(n, j0 + i + w + 1 - i0)):
            cost = min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
            matrix[i, j] = dist_mat[i, j] + cost

    if backtr_loc is None:
        # find starting point of back-track
        loc = np.where(matrix == min(matrix[-1, :].min(), matrix[:, -1].min()))
        i, j = loc[0][-1], loc[1][-1]
    else: i,j = backtr_loc

    path = []

    while i > i0 or j > j0:
        path.append((i, j))

        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf

#        move = np.argmin([option_diag, option_up, option_left])
        a = (option_diag, option_up, option_left)
        move = a.index(min(a))

        if move == 2:
            j -= 1
        elif move == 1:
            i -= 1
        else:
            i -= 1
            j -= 1

    #    path.append((i0,j0))
    path.reverse()

    return path, matrix[-1, -1], matrix


@jit(nopython=True)
def LCMA_jit(A, L, extend_r=0.0, end_cut=0):
    # enc_cut: dont consider start and end of warp path

    cumsum = np.cumsum(A)
    i0 = 0
    j0 = len(cumsum) - 1
    min_mean = (cumsum[j0] - cumsum[i0]) / (j0 - i0)

    for i in range(end_cut,len(cumsum)-end_cut):
        for j in range(i + L, len(cumsum) -end_cut):
            tmp_mean = (cumsum[j] - cumsum[i]) / (j - i)
            if tmp_mean < min_mean:
                min_mean = tmp_mean
                i0 = i
                j0 = j

    if extend_r > 0:
        extension = (j0 - i0) * extend_r
        i0 = max(0, int(i0 - extension/2) )
        j0 = min(len(A)-1, int(j0 + extension/2) )
        min_mean = (cumsum[j0] - cumsum[i0]) / (j0 - i0)

    return i0, j0, min_mean



@jit(nopython=True)
def LCMA_jit_new(A, L, extend_r=0.0, end_cut=0):
    # enc_cut: dont consider start and end of warp path

    cumsum = np.cumsum(A)
    i0 = 0
    j0 = len(cumsum) - 1
    min_mean = (cumsum[j0] - cumsum[i0]) / (j0 - i0)

    for i in range(end_cut,len(cumsum)-end_cut):
        for j in range(i + L, len(cumsum) -end_cut):
            tmp_mean = (cumsum[j] - cumsum[i]) / (j - i)
            if tmp_mean < min_mean:
                min_mean = tmp_mean
                i0 = i
                j0 = j

    if extend_r > 0:
        extend_thr = min_mean*extend_r #min_mean * 1.01

        while (i0>1) and (A[i0] < extend_thr): i0 -=1
        while (j0<len(A)-1) and (A[j0] < extend_thr): j0 +=1

    return i0, j0, min_mean


@jit
def gen_seeds(mat_shape, w=8, overlap=False):
    if not overlap: w = 2 * w

    seeds = []
    for k in range(int(np.floor((mat_shape[0] - 1) / (w + 1)))): seeds.append((k * (w + 1), 0))
    for k in range(1, int(np.floor((mat_shape[1] - 1) / (w + 1)))): seeds.append((0, k * (w + 1)))

    return seeds


@jit
def sdtw_min_paths_jit(dist_mat, w, seeds):
    # returns best paths for each seed
    paths = []
    for seed in seeds:
        path, cost, matrix = sdtw_jit(dist_mat, w=w, start=seed)
        paths.append(path)

    return paths




from sklearn.metrics.pairwise import pairwise_distances

@jit
def angle_loss(a0, a1):
    """ smallest difference between two angles """

    degrees = True
    if degrees:
        max_angle = 360
    else:
        max_angle = np.pi * 2

    diff = abs(a0 - a1)

    return np.sum(np.minimum(max_angle - diff, diff))



def joints_loss(feats0, feats1, angle_w):
    """ :returns n0 x n1 dist mat  """

    idx = 4 # where angles end in feature array

    dist_mat_angles = pairwise_distances(feats0[:,:idx], feats1[:,:idx], metric=angle_loss) / 360
    dist_mat_lengths = pairwise_distances(feats0[:,idx:], feats1[:,idx:], metric='euclidean')

    dist_mat = dist_mat_lengths + angle_w * dist_mat_angles

    return dist_mat



