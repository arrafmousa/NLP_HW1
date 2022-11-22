import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test


def main():
    threshold = 1  # TODO : learn this parameter k-fld validation
    lam = 1

    train_path = "data/train1.wtag"
    train_path2 = "data/train1.wtag"
    test_path = "data/test1.wtag"
    comp_path1 = "data/comp1.words"
    comp_path2 = "data/comp2.words"

    weights_path = 'weights.pkl'
    predictions_path = 'predictions.wtag'
    predictions_com1_path = 'predictions_comp1.wtag'
    predictions_com2_path = 'predictions_comp2.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, statistics.tags)
    tag_all_test(comp_path1, pre_trained_weights, feature2id, predictions_com1_path, statistics.tags)

    statistics, feature2id = preprocess_train(train_path2, 10)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    tag_all_test(comp_path2, pre_trained_weights, feature2id, predictions_com2_path, statistics.tags)

if __name__ == '__main__':
    main()
