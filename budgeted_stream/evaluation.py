import numpy as np

# python version of https://github.com/zhunzhong07/person-re-ranking/blob/master/evaluation/Market_1501_evaluation.m

def get_ap(good_image, junk, index):
    ap = 0
    old_recall = 0
    old_precision = 1.0
    intersect_size = 0
    n_junk = 0
    ngood = len(good_image)
    good_now = 0
    j = 0
    for i in range(index.shape[0]):
        flag = False
        if index[i] in junk:
            n_junk += 1
            continue
        if index[i] in good_image:
            good_now += 1
            flag = True
        if flag:
            intersect_size = intersect_size + 1
        recall = intersect_size/ngood
        precision = intersect_size/(j + 1)
        ap = ap + (recall - old_recall)*((old_precision+precision)/2)
        old_recall = recall
        old_precision = precision
        j = j + 1
        if good_now == ngood:
            return ap
    return ap

def get_rank(good_image, junk, index):
    rank = index.shape[0]
    n_junk = 0
    for i in range(index.shape[0]):
        if index[i] in junk:
            n_junk += 1
            continue
        if index[i] in good_image:
            return i - n_junk

def get_good_junk(q, label_gallery, label_query, cam_gallery, cam_query, junk_0):
    q_label = label_query[q]
    q_cam = cam_query[q]
    pos = label_gallery == q_label
    pos_2 = cam_gallery != q_cam
    good_image = np.argwhere(np.logical_and(pos, pos_2))
    if good_image.shape[0] > 1:
        good_image = good_image.squeeze()
    else:
        good_image = good_image[0]
    pos_3 = cam_gallery == q_cam
    junk = np.argwhere(np.logical_and(pos, pos_3)).squeeze()
    junk = np.append(junk_0, junk)
    return good_image, junk

def evaluate(distance, label_gallery, label_query, cam_gallery, cam_query):
    mAP = []
    junk_0 = np.argwhere(label_gallery == "-1").squeeze()
    ranks = np.zeros(distance.shape[1])
    for q in range(distance.shape[0]):
        score = distance[q]
        good_image, junk = get_good_junk(q, label_gallery, label_query, cam_gallery, cam_query, junk_0)
        index = np.argsort(score)
        rank = get_rank(good_image, junk, index)
        ap = get_ap(good_image, junk, index)
        mAP.append(ap)
        ranks[rank] += 1
    cmc = []
    last_rank = 0
    mAP = np.mean(mAP)
    for i in range(distance.shape[1]):
        cmc.append((last_rank + ranks[i]) / distance.shape[0])
        last_rank += ranks[i]
    return cmc, mAP

