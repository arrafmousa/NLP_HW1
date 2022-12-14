from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple
from operator import itemgetter

WORD = 0
TAG = 1


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated
        # self.suffix_list = [
        #     "ee",
        #     "eer",
        #     "er",
        #     "ion",
        #     "ism",
        #     "ity",
        #     "ment",
        #     "ness",
        #     "or",
        #     "sion",
        #     "ship",
        #     "th",
        #     "ible",
        #     "able",
        #     "al",
        #     "ant",
        #     "ary",
        #     "ful",
        #     "ic",
        #     "ious",
        #     "ive",
        #     "less",
        #     "ous",
        #     "y",
        #     "ed",
        #     "en",
        #     "er",
        #     "ing",
        #     "ize",
        #     "ise",
        #     "ly",
        #     "ward",
        #     "wise",
        #     "s",  # TODO : check with and without if useful
        #     "es"
        # ]
        # self.prefix_list = [
        #     "an",
        #     "a",
        #     "ab",
        #     "ac",
        #     "as",
        #     "com",
        #     "ad",
        #     "ante",
        #     "anti",
        #     "auto",
        #     "ben",
        #     "bi",
        #     "circu",
        #     "counter",
        #     "contra",
        #     "con",
        #     "co",
        #     "de",
        #     "di",
        #     "dis",
        #     "e",
        #     "eu",
        #     "ex",
        #     "exo",
        #     "fore",
        #     "hemi",
        #     "hyper",
        #     "hypo",
        #     "il",
        #     "inter",
        #     "intra",
        #     "macro",
        #     "mal",
        #     "micro",
        #     "mis",
        #     "mono",
        #     "multi",
        #     "ecto",
        #     "extra",
        #     "extro",
        #     "im",
        #     "in",
        #     "ir",
        #     "non",
        #     "ob",
        #     "omni",
        #     "over",
        #     "peri",
        #     "poly",
        #     "post",
        #     "pre",
        #     "pro",
        #     "quad",
        #     "re",
        #     "semi",
        #     "sub",
        #     "super",
        #     "sym",
        #     "trans",
        #     "tri",
        #     "ultra",
        #     "un",
        #     "uni",
        #     "oc",
        #     "op",
        #     "sup",
        #     "sus",
        #     "syn",
        #     "supra"]
        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106",
                             "f107", "fCapital", "fAllCapital", "fNumeric", "fHyphen", "fContainsDigit", "fFirst",
                             "fLast"]  # the feature classes used in the code #TODO : fCapital and so on ..
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:  # f100
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)

    def get_suffix_tag_pair_count(self, file_path) -> None:  # f101
        """
            Extract out of text all suffix/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    for i in range(-4, -1):
                        cur_suffix = cur_word[i:]
                        if len(cur_word) > i:
                            # update the dict
                            if (cur_suffix, cur_tag) not in self.feature_rep_dict["f101"]:
                                self.feature_rep_dict["f101"][(cur_suffix, cur_tag)] = 1
                            else:
                                self.feature_rep_dict["f101"][(cur_suffix, cur_tag)] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)

    def get_prefix_tag_pair_count(self, file_path) -> None:  # f102
        """
                    Extract out of text all suffix/tag pairs
                    @param: file_path: full path of the file to read
                    Updates the histories list
                """
        with open(file_path) as file:
            for line in file:  # f107
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    for i in range(1, 5):
                        prefix = cur_word[:i]
                        if len(cur_word) > i:
                            if (prefix, cur_tag) not in self.feature_rep_dict["f102"]:
                                self.feature_rep_dict["f102"][(prefix, cur_tag)] = 1
                            else:
                                self.feature_rep_dict["f102"][(prefix, cur_tag)] += 1
                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)

    def get_k_wise_tag_count(self, file_path) -> None:  # f103-105
        """
            Extract out of text all Uni/bi/Tri -grams
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')

                uni_gram = [0]
                bi_gram = [0, 1]
                tri_gram = [0, 1, 2]

                while uni_gram[0] < len(split_words):

                    cur_tag = split_words[uni_gram[0]].split('_')[1]
                    if cur_tag not in self.feature_rep_dict["f105"]:
                        self.feature_rep_dict["f105"][cur_tag] = 1
                    else:
                        self.feature_rep_dict["f105"][cur_tag] += 1

                    if bi_gram[1] < len(split_words):
                        cur_tag = (split_words[bi_gram[0]].split('_')[1], split_words[bi_gram[1]].split('_')[1])
                        if cur_tag not in self.feature_rep_dict["f104"]:
                            self.feature_rep_dict["f104"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f104"][cur_tag] += 1
                    if tri_gram[2] < len(split_words):
                        cur_tag = (split_words[tri_gram[0]].split('_')[1], split_words[tri_gram[1]].split('_')[1],
                                   split_words[tri_gram[2]].split('_')[1])
                        if cur_tag not in self.feature_rep_dict["f103"]:
                            self.feature_rep_dict["f103"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f103"][cur_tag] += 1

                    uni_gram = [i + 1 for i in uni_gram]
                    bi_gram = [i + 1 for i in bi_gram]
                    tri_gram = [i + 1 for i in tri_gram]

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1],
                        sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                self.histories.append(history)

    def get_previous_word_current_tag_count(self, file_path) -> None:  # f106
        """
            Extract out of text all suffix/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    _, cur_tag = split_words[word_idx].split('_')
                    prev_word, _ = split_words[word_idx - 1].split('_')

                    if (prev_word, cur_tag) not in self.feature_rep_dict["f106"]:
                        self.feature_rep_dict["f106"][(prev_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f106"][(prev_word, cur_tag)] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)

    def get_next_word_current_tag_count(self, file_path) -> None:  # f107
        """
            Extract out of text all suffix/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words) - 1):
                    _, cur_tag = split_words[word_idx].split('_')
                    next_word, _ = split_words[word_idx + 1].split('_')

                    if (next_word, cur_tag) not in self.feature_rep_dict["f107"]:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)

    def get_features_tag_count(self, file_path) -> None:  # all other features
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')

                    # fCapital
                    if cur_word[0].isupper():
                        if cur_tag not in self.feature_rep_dict["fCapital"]:
                            self.feature_rep_dict["fCapital"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["fCapital"][cur_tag] += 1
                    # fAllCapital
                    if cur_word.isupper():
                        if cur_tag not in self.feature_rep_dict["fAllCapital"]:
                            self.feature_rep_dict["fAllCapital"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["fAllCapital"][cur_tag] += 1
                    # fNumeric
                    if cur_word.isnumeric():
                        if cur_tag not in self.feature_rep_dict["fNumeric"]:
                            self.feature_rep_dict["fNumeric"][cur_tag] = 1
                            #print("new " + cur_tag)
                        else:
                            self.feature_rep_dict["fNumeric"][cur_tag] += 1
                            # print("added to " + cur_tag)
                    # fHyphen
                    if cur_word.find('-') != -1:
                        if cur_tag not in self.feature_rep_dict["fHyphen"]:
                            self.feature_rep_dict["fHyphen"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["fHyphen"][cur_tag] += 1
                    # fContainsDigit
                    if any(chr.isdigit() for chr in cur_word) != -1:
                        if cur_tag not in self.feature_rep_dict["fContainsDigit"]:
                            self.feature_rep_dict["fContainsDigit"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["fContainsDigit"][cur_tag] += 1
                    # fFirst
                    if word_idx == 0:
                        if cur_tag not in self.feature_rep_dict["fFirst"]:
                            self.feature_rep_dict["fFirst"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["fFirst"][cur_tag] += 1
                    # fLast
                    if word_idx == len(split_words)-1:
                        if cur_tag not in self.feature_rep_dict["fLast"]:
                            self.feature_rep_dict["fLast"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["fLast"][cur_tag] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {
            "f100": OrderedDict(),
            "f101": OrderedDict(),
            "f102": OrderedDict(),
            "f103": OrderedDict(),
            "f104": OrderedDict(),
            "f105": OrderedDict(),
            "f106": OrderedDict(),
            "f107": OrderedDict(),
            "fCapital": OrderedDict(),
            "fAllCapital": OrderedDict(),
            "fNumeric": OrderedDict(),
            "fHyphen": OrderedDict(),
            "fContainsDigit": OrderedDict(),
            "fFirst": OrderedDict(),
            "fLast": OrderedDict(),
        }
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) \
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word = history[0]
    c_tag = history[1]
    p_word = history[2]
    p_tag = history[3]
    pp_word = history[4]
    pp_tag = history[5]
    n_word = history[6]  # next word
    features = []

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    # f101
    for i in range(-4, -1):
        suffix = c_word[i:]
        if len(c_word) > i:
            if (suffix, c_tag) in dict_of_dicts["f101"]:
                features.append(dict_of_dicts["f101"][(suffix, c_tag)])

    # f102
    for i in range(1, 5):
        prefix = c_word[:i]
        if len(c_word) > i:
            if (prefix, c_tag) in dict_of_dicts["f102"]:
                features.append(dict_of_dicts["f102"][(prefix, c_tag)])

    # f103
    if (pp_tag, p_tag, c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(pp_tag, p_tag, c_tag)])

    # f104
    if (p_tag, c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(p_tag, c_tag)])

    # f105
    if c_tag in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][c_tag])

    # f106
    if (p_word, c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(p_word, c_tag)])

    # f107
    if (n_word, c_tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(n_word, c_tag)])

    # fCapital
    if str(c_word[0]).isupper():
        try:
            features.append(dict_of_dicts["fCapital"][c_tag])
        except:
            pass

    # fAllCapital
    if c_word.isupper():
        if c_tag in dict_of_dicts["fAllCapital"]:
            features.append(dict_of_dicts["fAllCapital"][c_tag])

    # fNumeric
    if str(c_word).isnumeric():
        try:
            features.append(dict_of_dicts["fNumeric"][c_tag])
        except:
            pass

    # fHyphen
    if c_word.find('-') != -1:
        if c_tag in dict_of_dicts["fHyphen"]:
            features.append(dict_of_dicts["fHyphen"][c_tag])

    #fContainsDigit
    if any(chr.isdigit() for chr in c_word):
       if c_tag in dict_of_dicts["fContainsDigit"]:
           features.append(dict_of_dicts["fContainsDigit"][c_tag])

    # fFirst
    if p_word == '*':
        if c_tag in dict_of_dicts["fFirst"]:
            features.append(dict_of_dicts["fFirst"][c_tag])

    # fLast
    if n_word == '~':
        if c_tag in dict_of_dicts["fLast"]:
            features.append(dict_of_dicts["fLast"][c_tag])

    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)
    statistics.get_suffix_tag_pair_count(train_path)
    statistics.get_prefix_tag_pair_count(train_path)
    statistics.get_k_wise_tag_count(train_path)
    statistics.get_next_word_current_tag_count(train_path)
    statistics.get_previous_word_current_tag_count(train_path)
    statistics.get_features_tag_count(train_path)
    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
