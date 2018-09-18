import argparse
import numpy as np
import os
import re
import os.path as osp
try:
    import _pickle as pickle
except:
    import pickle
import scipy.io

from confidence_functions import max_neg_dist_function, margin_function
from src.utils import get_colored_logger

STAGES = 4

def get_eu_distance(query_feats, gallery_feats):
    norm_q = (query_feats * query_feats).sum(axis=1)
    norm_q = np.tile(norm_q, (gallery_feats.shape[0], 1)).T
    norm_g = (gallery_feats * gallery_feats).sum(axis=1)
    norm_g = np.tile(norm_g, (query_feats.shape[0], 1))
    quick_dist = norm_q + norm_g - 2. * query_feats.dot(gallery_feats.T)

    return quick_dist

def get_junk(q, label_gallery, label_query, cam_gallery, cam_query):
    q_label = label_query[q]
    q_cam = cam_query[q]
    pos = label_gallery == q_label
    pos_3 = cam_gallery == q_cam
    junk = np.logical_and(pos, pos_3)
    return junk

def get_threshold_offline(p, confidences):
    theta = []
    remains = np.ones(len(confidences[0]), dtype=bool)
    total = remains.sum()
    left_prob = 1
    # print(remains.sum())
    for i in range(0, STAGES - 1):
        if remains.sum() == 0 or p[i] / left_prob <= 1e-6:
            theta.append(1)
            continue
        sorted_confidence = np.sort(confidences[i][remains])
        theta.append(sorted_confidence[-int(len(sorted_confidence) * 1.0 * p[i] / left_prob) - 1])
        new_remain = confidences[i] < theta[-1]
        exit = np.logical_and(confidences[i] >= theta[-1], remains)
        # print(exit.sum() * 1.0 / remains.sum())
        # print(exit.sum() * 1.0 / total)
        # print("exit: {}, remain: {}, frac: {}".format(exit.sum(), np.logical_and(remains, new_remain).sum(), exit.sum() * 1.0 / total))
        # print("threshold: {}".format(theta[-1]))
        remains = np.logical_and(remains, new_remain)
        left_prob -= p[i]

    return theta

def gen_val_confidence(validation_images_datapath, extracted_feature_datapath, confidence_function="distance"):
    logger = get_colored_logger("exp")
    STAGES = 4

    image_list = [f for f in os.listdir(validation_images_datapath) if f.endswith(".jpg") and not f.startswith("-1")]
    image_list.sort()

    persons = []
    pattern = r'(?P<person>\d{4})_c\ds\d_\d{6}_\d{2}.jpg'
    for i in image_list:
        match = re.match(pattern, i)
        if match:
            persons.append(match.group("person"))
    persons = list(set(persons))
    persons.sort()
    logger.info(">> total val person: {}".format(len(persons)))


    ## Read in validation feature vectors on 4 different stages
    val_features = []
    for i in range(STAGES-1):
        val_features.append(scipy.io.loadmat(osp.join(extracted_feature_datapath, "train_features_{}.mat".format(i)))['feature_train_new'].T)
    val_features.append(scipy.io.loadmat(osp.join(extracted_feature_datapath, "train_features_fusion.mat"))['feature_train_new'].T)

    ## Seperate the query and gallery set
    pattern = r'(?P<person>\d{4})_c(?P<cam>\d)s\d_\d{6}_\d{2}.jpg'
    label_person = np.array([re.match(pattern, i).group("person") for i in image_list])
    label_cam = np.array([re.match(pattern, i).group("cam") for i in image_list])

    query_idx = []
    delete_person = []
    for p in persons:
        locate_person = label_person == p
        cameras = list(set(label_cam[locate_person]))
        cameras.sort()
        flag = False
        temp_query = []
        for cam in cameras:
            locate_camera = label_cam == cam
            if len(np.argwhere(np.logical_and(locate_person, locate_camera))) > 1:
                flag = True
            locate_p_c = np.argwhere(np.logical_and(locate_person, locate_camera))[0][0]
            temp_query.append(locate_p_c)
        if flag:
            query_idx += temp_query
        else:
            logger.info(">> delete person {}".format(p))
            delete_person.append(p)
    query_idx = np.array(query_idx)
    gallery_idx = np.array([i for i in range(len(image_list)) if (not i in query_idx) and (label_person[i] not in delete_person)])

    query_feats = []
    gallery_feats = []
    for i in range(STAGES):
        query_feats.append(val_features[i][query_idx])
        gallery_feats.append(val_features[i][gallery_idx])

    label_query = label_person[query_idx]
    label_query_cam = label_cam[query_idx]
    label_gallery = label_person[gallery_idx]
    label_gallery_cam = label_cam[gallery_idx]
    for p in delete_person:
        persons.remove(p)
    logger.info(">> query size {}".format(len(query_idx)))
    logger.info(">> gallery size {}".format(len(gallery_idx)))

    ## generate the confidence for the query set for all four stages
    confidences = []
    for stage in range(STAGES):
        distance = np.sqrt(get_eu_distance(query_feats[stage], gallery_feats[stage]))
        confidence = np.zeros(len(label_query))
        for q in range(len(label_query)):
            junk = get_junk(q, label_gallery, label_query, label_gallery_cam, label_query_cam)
            # print(junk.sum())
            valid_idx = np.logical_not(junk)
            valid_distance = distance[q][valid_idx]
            if confidence_function == "distance":
                confidence[q] = max_neg_dist_function(valid_distance)
            elif confidence_function == "margin":
                confidence[q] = margin_function(valid_distance, label_gallery[valid_idx])
        confidences.append(confidence)

    return confidences

    # pickle.dump(file=open(osp.join(save_path, "infos.pkl"), "wb"), obj={"num_query": len(query_idx)})