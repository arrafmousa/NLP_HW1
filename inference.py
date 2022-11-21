import itertools

from preprocessing import read_test, represent_input_with_features, Feature2id, FeatureStatistics
from tqdm import tqdm
import math
import numpy as np
from numpy import unravel_index


def calculate_probability(sentence, prev_tagging, index, v, feature2id: Feature2id, tags):
    """

    :param sentence:
    :param prev_tagging:
    :param index:
    :param v:
    :param feature2id:
    :param tags:
    :return:
    """

    exp_scores = {}
    exp_scores_sum = 0
    # calculate score from v and features vector
    for y in tags:
        x_y = sentence[index], y, sentence[index - 1], prev_tagging[1], sentence[index - 2], prev_tagging[0] \
            , sentence[index + 1]
        f_vector = represent_input_with_features(x_y, feature2id.feature_to_idx)
        score = 0
        for idx in f_vector:
            score += v[idx]
        exp_score = math.exp(score)
        exp_scores[y] = exp_score
        exp_scores_sum += exp_score

    probs = {}
    # calculate probabilities
    for y in tags:
        probs[y] = exp_scores[y] / exp_scores_sum

    return probs


def memm_viterbi(sentence, pre_trained_weights, feature2id, tags, beam_k=None):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    tgs_idx = list(tags)
    tgs_idx.append("*")
    n = len(sentence)
    # pi [k,u,v] = what is the maximum probability of the sentence with length k to end in the labels u, v
    pi = np.zeros((n, len(tgs_idx), len(tgs_idx)))
    bp = np.zeros((n, len(tgs_idx), len(tgs_idx)))

    # initialize
    for u in range(len(tgs_idx)):
        for v in range(len(tgs_idx)):
            if tgs_idx[u] == "*" and tgs_idx[v] == "*":
                pi[0][u][v] = 1
            else:
                pi[0][u][v] = 0

    # inductive step / with cropping up to k
    for k in range(1, len(sentence)):
        for u in range(len(tgs_idx)):  # w u v
            for t in range(len(tgs_idx)):
                # find max and argmax
                max_prob = -1
                max_label = None
                if pi[k - 1][t][u] == 0:
                    continue
                else:
                    probs = calculate_probability(sentence, [tgs_idx[t], tgs_idx[u]], k,
                                                  pre_trained_weights,
                                                  feature2id, tgs_idx)
                    for v in range(len(tgs_idx)):
                        prob_v = probs.get(tgs_idx[v]) if tgs_idx[v] in probs.keys() else 0
                        prob_uv_given_t = pi[k - 1][t][u] * prob_v
                        if prob_uv_given_t > pi[k][u][v]:
                            pi[k][u][v] = prob_uv_given_t
                            bp[k][u][v] = t

        # beam cropping
        if beam_k is not None:
            idx = np.argpartition(pi[k].flatten(), -beam_k)
            threshold = min(pi[k].flatten()[idx[-beam_k:]])
            pi[k][pi[k][:][:] < threshold] = 0

    pred_u, pred_v = unravel_index(pi[0].argmax(), pi[0].shape)
    predictions = []
    for word in range(0, len(sentence)):
        pred_u, pred_v = pred_v, pi[word][pred_u].argmax()
        print("the word " + str(sentence[word]) + " was tagged " + str(tgs_idx[pred_u]))

    return predictions


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, tags):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, tags, 5)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
