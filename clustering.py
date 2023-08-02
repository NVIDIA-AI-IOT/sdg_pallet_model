import torch
import numpy as np
from sklearn.cluster import DBSCAN

import time
import numpy as np


IDENTITY_PERM = np.array([0, 1, 2, 3, 4, 5, 6, 7])
ROTATE_Z_90_PERM = [1, 3, 0, 2, 5, 7, 4, 6]
ROTATE_Y_180_PERM = [5, 4, 7, 6, 1, 0, 3, 2]
MIRROR_X_PERM = [1, 0, 3, 2, 5, 4, 7, 6]

PALLET_PERMUTATIONS = [
    IDENTITY_PERM,
    IDENTITY_PERM[ROTATE_Z_90_PERM],
    IDENTITY_PERM[ROTATE_Z_90_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[ROTATE_Z_90_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[ROTATE_Y_180_PERM],
    IDENTITY_PERM[ROTATE_Y_180_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[ROTATE_Y_180_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[ROTATE_Y_180_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM],
]

PALLET_PERMUTATIONS_WITH_MIRRORS = [
    IDENTITY_PERM[MIRROR_X_PERM],
    IDENTITY_PERM[MIRROR_X_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[MIRROR_X_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[MIRROR_X_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[MIRROR_X_PERM][ROTATE_Y_180_PERM],
    IDENTITY_PERM[MIRROR_X_PERM][ROTATE_Y_180_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[MIRROR_X_PERM][ROTATE_Y_180_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM],
    IDENTITY_PERM[MIRROR_X_PERM][ROTATE_Y_180_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM][ROTATE_Z_90_PERM],
]


UPRIGHT_PALLET_SIDE_FACE_INDICES = [0, 1, 5, 4]
UPRIGHT_PALLET_PERMUTATIONS = [
    PALLET_PERMUTATIONS[0],
    PALLET_PERMUTATIONS[1],
    PALLET_PERMUTATIONS[2],
    PALLET_PERMUTATIONS[3]
]

BOX_EDGES = [
    [0, 1],
    [1, 5],
    [5, 4],
    [4, 0],
    [2, 3],
    [3, 7],
    [7, 6],
    [6, 2],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7]
]

UPRIGHT_PALLET_SIDE_FACE_INDICES_ALL = [perm[UPRIGHT_PALLET_SIDE_FACE_INDICES] for perm in UPRIGHT_PALLET_PERMUTATIONS]


class Clustering(object):

    def __init__(self, eps=10, min_samples=1, permutations=UPRIGHT_PALLET_PERMUTATIONS):
        self.clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        self.permutations = permutations

    def compute_distance_matrix(self, kps):
        kps_a = kps[:, None, None, :, :]
        kps_b = kps[None, :, self.permutations, :]

        kps_ab_dist, _ = torch.min(torch.mean(torch.sqrt(torch.sum((kps_a - kps_b)**2, dim=-1)), dim=-1), dim=-1)
        distance_matrix = kps_ab_dist.detach().cpu().numpy()
        return distance_matrix

    def cluster(self, kps):
        distance_matrix = self.compute_distance_matrix(kps)
        labels = self.clustering.fit_predict(distance_matrix)
        clusters = np.unique(labels)

        perm_kps = kps[:, self.permutations, :]

        nms_boxes = []

        for cluster in clusters:

            indices = np.nonzero(labels == cluster)[0]

            ref_box_idx = indices[0]
            ref_box = kps[ref_box_idx]

            clust_boxes = []
            clust_boxes.append(ref_box)
            for i in range(1, len(indices)):
                box_idx = indices[i]
                targ_box = perm_kps[box_idx]
                dist, min_idx = torch.min(torch.mean(torch.sqrt(torch.sum((ref_box - targ_box)**2, dim=-1)), dim=-1), dim=-1)
                box_perm_match = targ_box[min_idx]
                clust_boxes.append(box_perm_match)

            clust_box = torch.mean(torch.stack(clust_boxes, dim=0), dim=0)
            nms_boxes.append(clust_box)

        nms_boxes = torch.stack(nms_boxes, dim=0)

        return nms_boxes