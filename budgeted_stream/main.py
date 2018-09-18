import numpy as np
import os
import re
import os.path as osp
try:
    import _pickle as pickle
except:
    import pickle
import scipy.io
import argparse
from utils.logging import add_file_handle, set_colored_logger
from evaluation import evaluate
from confidence_functions import max_neg_dist_function, margin_function

STAGES = 4

# flops (unit: macc) are calculated using http://dgschwend.github.io/netscope
FLOPs = [523568640, 1150094336, 2060915200, 2542295040]

def get_eu_distance(query_feats, gallery_feats):
    norm_q = (query_feats * query_feats).sum(axis=1)
    norm_q = np.tile(norm_q, (gallery_feats.shape[0], 1)).T
    norm_g = (gallery_feats * gallery_feats).sum(axis=1)
    norm_g = np.tile(norm_g, (query_feats.shape[0], 1))
    quick_dist = norm_q + norm_g - 2. * query_feats.dot(gallery_feats.T)

    return quick_dist

def get_p_given_budget(budget):
    assert budget >= FLOPs[0], "budget invalid"
    coeff = []
    for s in range(STAGES):
        coeff.append(FLOPs[STAGES - 1 - s] - budget)
    roots = np.roots(coeff)
    q = roots.real[abs(roots.imag) < 1e-5][0]
    p = []
    for i in range(STAGES):
        p.append(q ** i)
    total = np.sum(p)
    p = [element * 1.0 / total for element in p]
    return p

def get_p_given_budget_random(budget):
    assert budget >= FLOPs[0], "budget invalid"
    p = []
    # find the slot
    slot = None
    for s in range(STAGES - 1):
        if budget >= FLOPs[s] and budget < FLOPs[s + 1]:
            slot = s
    if slot is None:
        # here budget >= FLOPs[-1], so just use the last stage
        p = [0, 0, 0, 1]
    else:
        for s in range(slot):
            p.append(0)
        p.append((FLOPs[slot + 1] - budget) / (FLOPs[slot + 1] - FLOPs[slot]))
        p.append(1 - (FLOPs[slot + 1] - budget) / (FLOPs[slot + 1] - FLOPs[slot]))
        for s in range(slot + 2, STAGES):
            p.append(0)
    return p

def get_budget_given_p(p):
    count = 0
    for s in range(STAGES):
        count += FLOPs[s] * p[s]
    return count

def parse_args():
    parser = argparse.ArgumentParser(description="Simulation on budgeted stream person re-ID scenarios")

    parser.add_argument("--dataset_path",
                        help="root path Market1501 dataset", type=str,
                        default="./dataset/Market-1501-v15.09.15")

    parser.add_argument("--log_file",
                        help="log file name", type=str,
                        default="test_log")

    parser.add_argument("--feature_path",
                        help="root datapath for features matrices of different stages", type=str,
                        default="./data/feature/DaRe")

    parser.add_argument("--save_path",
                        help="path for saving new distance matrices", type=str,
                        default="./results/DaRe")

    parser.add_argument("--q_list",
                        help="q list for testing budgets", type=float, nargs="+",
                        default=[0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.05, 1.10, 1.50, 2.00, 3.00, 5.00, 10.00, 50.00])

    parser.add_argument("--budget_list",
                        help="budget list for testing budgets", type=float, nargs="+",
                        default=np.linspace(FLOPs[0], FLOPs[-1], 50).tolist())

    parser.add_argument("--seed", help="set random seed (+1s!)",
                        default=817, type=int)

    parser.add_argument("--test_budget", dest="test_q",
                        help="input budget, test performance; otherwise input q, test performance",
                        action="store_false")

    parser.add_argument("--dump_exit_history", dest="dump_exit_history",
                        help="toggle to dump exit history for each q (for figures in section 5.4 (Qualitative Results) and supplementary)",
                        action="store_true")

    parser.add_argument("--dump_distance_mat", dest="dump_distance_mat",
                        help="toggle to dump resulted distance matrix so that you can use official matlab code to do the evaluation (indeed will get the same results as using our python implementation). URL: https://github.com/zhunzhong07/person-re-ranking/blob/master/evaluation/Market_1501_evaluation.m",
                        action="store_true")

    parser.add_argument("--confidence_function",
                        help="what confidence function to use (<margin|distance>)",
                        type=str, choices=["margin", "distance", "random"], default="distance")


    return parser.parse_args()

def get_junk(q, label_gallery, label_query, cam_gallery, cam_query):
    q_label = label_query[q]
    q_cam = cam_query[q]
    pos = label_gallery == q_label
    pos_3 = cam_gallery == q_cam
    junk = np.logical_and(pos, pos_3)
    return junk

def gen_exit_stage(p, distances,
                          image_list_query,
                          label_person_gallery,
                          label_person_query,
                          label_cam_gallery,
                          label_cam_query,
                          confidence_function):

    exit_stage = np.zeros(len(image_list_query), dtype=int)
    exit_retrived = np.zeros(len(image_list_query), dtype=int)
    exit_confidence = np.zeros(len(image_list_query), dtype=float)
    remains = np.ones(len(image_list_query), dtype=bool)

    exit_prob = []
    partial_sum = 0
    for stage in range(len(p)):
        if 1 - partial_sum < 1e-6:
            exit_prob.append(1)
            continue
        exit_prob.append(p[stage] / (1 - partial_sum))
        partial_sum += p[stage]

    def query_exit(q, confidence, stage, retrived_id):
        exit_stage[q] = stage
        remains[q] = False
        exit_confidence[q] = confidence
        exit_retrived[q] = retrived_id

    confidences = []
    for stage in range(STAGES - 1):
        confidences.append([])
    junk_0 = label_person_gallery == '-1'
    valid_0 = np.logical_not(junk_0)
    for q in range(len(image_list_query)):
        for stage in range(STAGES):
            junk = get_junk(q, label_person_gallery, label_person_query, label_cam_gallery, label_cam_query)
            valid_idx = valid_0.copy()
            valid_idx = np.logical_and(valid_idx, np.logical_not(junk))
            valid_distance = distances[stage][q][valid_idx]
            ids = np.argwhere(valid_idx)
            retrived_id = ids[np.argmax(-valid_distance)][0]

            if confidence_function == "random":
                if np.random.rand() <= exit_prob[stage]:
                    query_exit(q, 0, stage, retrived_id)
                    break
            else:
                if confidence_function == "distance":
                    confidence = max_neg_dist_function(valid_distance)
                elif confidence_function == "margin":
                    confidence = margin_function(valid_distance, label_person_gallery[valid_idx])
                if stage == STAGES - 1:
                    query_exit(q, confidence, stage, retrived_id)
                    break

                if len(confidences[stage]) == 0:
                    confidences[stage].append(confidence)
                    query_exit(q, confidence, stage, retrived_id)
                    break
                else:
                    confidences[stage].append(confidence)
                    rank_list = np.sort(confidences[stage])[::-1]
                    threshold = rank_list[int((len(rank_list) - 1) * exit_prob[stage])]
                    if confidence >= threshold:
                        query_exit(q, confidence, stage, retrived_id)
                        break

    assert remains.sum() == 0
    return exit_stage, exit_retrived, exit_confidence

def load_labels(args):
    query_images_datapath = osp.join(args.dataset_path, "query")
    gallery_images_datapath = osp.join(args.dataset_path, "bounding_box_test")
    image_list_query = [f for f in os.listdir(query_images_datapath) if f.endswith(".jpg")]
    image_list_query.sort()
    image_list_gallery = [f for f in os.listdir(gallery_images_datapath) if f.endswith(".jpg")]
    image_list_gallery.sort()
    pattern = r'(?P<person>\d{4}|-1)_c(?P<cam>\d)s\d_\d{6}_\d{2}.jpg'

    label_person_gallery = np.array([re.match(pattern, i).group("person") for i in image_list_gallery])
    label_cam_gallery = np.array([re.match(pattern, i).group("cam") for i in image_list_gallery])
    label_person_query = np.array([re.match(pattern, i).group("person") for i in image_list_query])
    label_cam_query = np.array([re.match(pattern, i).group("cam") for i in image_list_query])

    return image_list_query, image_list_gallery, label_person_gallery, label_cam_gallery, label_person_query, label_cam_query

def load_distances(args):
    distances = []
    for s in range(1, STAGES):
        query = scipy.io.loadmat(osp.join(args.feature_path, "query_features_{}.mat".format(s)))['feature_query_new'].T
        gallery = scipy.io.loadmat(osp.join(args.feature_path, "test_features_{}.mat".format(s)))['feature_test_new'].T
        distance = np.sqrt(get_eu_distance(query, gallery))
        distances.append(distance)
    query = scipy.io.loadmat(osp.join(args.feature_path, "query_features_fusion.mat"))['feature_query_new'].T
    gallery = scipy.io.loadmat(osp.join(args.feature_path, "test_features_fusion.mat"))['feature_test_new'].T
    distance = np.sqrt(get_eu_distance(query, gallery))
    distances.append(distance)
    return distances

def test(args):
    logger = args.logger
    # load labels
    image_list_query, image_list_gallery, label_person_gallery, label_cam_gallery, label_person_query, label_cam_query = load_labels(args)

    # load distance matrices
    logger.info("loading data")
    distances = load_distances(args)
    logger.info("finished loading data")

    if args.dump_exit_history:
        if not osp.isdir(osp.join(args.save_path, "exit_history")):
            os.makedirs(osp.join(args.save_path, "exit_history"))

    # simulation infos
    if args.confidence_function == "random":
        args.test_q = False
    logger.info("-> confidence function: {}".format(args.confidence_function))
    if args.test_q:
        logger.info("-> test on q = {}".format(", ".join(["{:.2f}".format(item) for item in args.q_list])))
    else:
        logger.info("-> test on budget = {}".format(", ".join(["{:.2f}".format(item) for item in args.budget_list])))

    # simulation begins
    logger.info("-> stream budgeted simulation begins")

    CMCs = []
    expected_budgets = []
    resulted_budgets = []
    num_case = len(args.q_list) if args.test_q else len(args.budget_list)
    for kase in range(num_case):
        if args.confidence_function == "random":
            budget = args.budget_list[kase]
            p = get_p_given_budget_random(budget)
        else:
            if args.test_q:
                q = args.q_list[kase]
                p = []
                for i in range(STAGES):
                    p.append(q ** i)
                total = np.sum(p)
                p = [element * 1.0 / total for element in p]
                budget = get_budget_given_p(p)
            else:
                budget = args.budget_list[kase]
                if budget < FLOPs[-1]:
                    p = get_p_given_budget(budget)
                else:
                    # special treatment, where q will be inf when p = p = [0, 0, 0, 1]
                    p = [0, 0, 0, 1]
        expected_budgets.append(budget)
        logger.info(">> test case {}".format(kase))
        if args.test_q:
            logger.info("   q = {}".format(q))
        logger.info("   expected p = {}".format(", ".join(["{:.3%}".format(item) for item in p])))
        logger.info("   expected average budget = {:.2f} macc".format(budget))
        exit_stage, exit_retrived, exit_confidence = gen_exit_stage(p=p, distances=distances,
                                                                           image_list_query=image_list_query,
                                                                           label_person_gallery=label_person_gallery,
                                                                           label_person_query=label_person_query,
                                                                           label_cam_gallery=label_cam_gallery,
                                                                           label_cam_query=label_cam_query,
                                                                           confidence_function=args.confidence_function)

        result_p = []
        for s in range(STAGES):
            result_p.append((exit_stage == s).sum() * 1.0 / len(exit_stage))
        resulted_budgets.append(get_budget_given_p(result_p))
        logger.info("   resulted average budget = {:.2f} macc".format(get_budget_given_p(result_p)))
        logger.info("   resulted p = {}".format(", ".join(["{:.3%}".format(item) for item in result_p])))

        distance_eu = np.zeros((len(image_list_query), len(image_list_gallery)))
        for query in range(len(image_list_query)):
            exit = exit_stage[query]
            distance_eu[query] = distances[exit][query]

        cmc, mAP = evaluate(distance_eu, label_person_gallery, label_person_query, label_cam_gallery, label_cam_query)
        logger.info("   CMC rank-1 = {:.3%}, mAP = {:.3%}".format(cmc[0], mAP))
        CMCs.append(cmc[0])

        for s in range(STAGES):
            stage = s + 1
            if s == STAGES - 1:
                stage = "f"
            logger.info("      exit on stage {}: {:.3%} ({} / {})".format(stage, (exit_stage == s).sum() * 1.0 / len(exit_stage), (exit_stage == s).sum(), len(exit_stage)))

        if args.dump_exit_history:
            pickle.dump(file=open(osp.join(args.save_path, "exit_history", "exit_stage_q_{:.2f}.pkl".format(q)), "wb"), obj=exit_stage)
            pickle.dump(file=open(osp.join(args.save_path, "exit_history", "exit_retrived_q_{:.2f}.pkl".format(q)), "wb"), obj=exit_retrived)
            pickle.dump(file=open(osp.join(args.save_path, "exit_history", "exit_confidence_q_{:.2f}.pkl".format(q)), "wb"), obj=exit_confidence)

        if args.dump_distance_mat:
            scipy.io.savemat(os.path.join(args.save_path, "q_{:.2f}.mat".format(q)), {'distance_eu': distance_eu})

    errors = []
    for i in range(len(expected_budgets)):
        errors.append((resulted_budgets[i] - expected_budgets[i]) / expected_budgets[i])
    errors = np.mean(errors)
    logger.info("CMC rank-1: {}".format(", ".join(["{:.3%}".format(item) for item in CMCs])))
    logger.info("expected average budgets: {}".format(", ".join(["{:.2f}".format(item) for item in expected_budgets])))
    logger.info("resulted average budgets: {}".format(", ".join(["{:.2f}".format(item) for item in resulted_budgets])))
    logger.info("average budget misalign rate: {:.3%}".format(errors))
    pickle.dump(file=open(osp.join(args.save_path, args.log_file + "_info.pkl"), "wb"), obj={"CMCs": CMCs, "expected_budgets": expected_budgets, "resulted_budgets": resulted_budgets})

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    if not osp.isdir(args.save_path):
        os.makedirs(args.save_path)
    args.logger = set_colored_logger("exp", level="INFO")
    add_file_handle(args.logger, osp.join(args.save_path, args.log_file + ".log"))
    args.logger.info("results will be saved to {}".format(osp.abspath(args.save_path)))
    args.logger.info("logger file: {}".format(args.log_file + ".log"))
    test(args)