import pickle

import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test

comp_path1 = "data/comp1.words"
comp_path2 = "data/comp2.words"
train_path1 = "data/train1.wtag"
train_path2 = "data/train2.wtag"
threshold = 1
lam = 1
weights_path = 'weights.pkl'

statistics, feature2id = preprocess_train(train_path1, threshold)
get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
with open(weights_path, 'rb') as f:
    optimal_params, feature2id = pickle.load(f)
pre_trained_weights = optimal_params[0]

tag_all_test(comp_path1, pre_trained_weights, feature2id, "comp_m1_207571258_212701239.etag", statistics.tags)

statistics, feature2id = preprocess_train(train_path2, threshold)
get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
with open(weights_path, 'rb') as f:
    optimal_params, feature2id = pickle.load(f)
pre_trained_weights = optimal_params[0]

tag_all_test(comp_path2, pre_trained_weights, feature2id, "comp_m1_207571258_212701239.etag", statistics.tags)
