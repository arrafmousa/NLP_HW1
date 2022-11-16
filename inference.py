from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import math


def calculate_probability(sentence, tags, index, v, feature2id: Feature2id, featureStatistics: FeatureStatistics) -> dict:
    """
    :returns dictionary contain all possible tags as key with their respective predicted probability
    :param sentence: list containing the words in the sentence
    :param tags: the predicted tags of the past words in the sentence (<index)
    :param index: the index of the word which we need to calculate the probability of the tags
    :param v: the weights vector
    """

    exp_scores = {}
    exp_scores_sum = 0
    # calculate score from v and features vector
    for y in featureStatistics.tags:
        x_y = sentence[i], y, sentence[i - 1], tags[i - 1], sentence[i - 2], tags[i - 2], sentence[i + 1]
        f_vector = represent_input_with_features(x_y, feature2id.feature_to_idx)
        score = 0
        for idx in f_vector:
            score += v[idx]
        exp_score = math.exp(score)
        exp_scores[y] = exp_score
        exp_scores_sum += exp_score


    probs = {}
    # calculate probabilities
    for y in featureStatistics.tags:
        probs[y] = exp_scores[y] / exp_scores_sum

    return probs

def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    pass


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
