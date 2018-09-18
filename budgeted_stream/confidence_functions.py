import numpy as np

def max_neg_dist_function(dist):
    return np.max(-dist)

def margin_function(dist, label):
    first_id = label[np.argmin(dist)]
    first_distance = np.min(dist)
    valid_idx = label != first_id
    valid_distance = dist[valid_idx]
    second_distance = np.min(valid_distance)
    return second_distance - first_distance
