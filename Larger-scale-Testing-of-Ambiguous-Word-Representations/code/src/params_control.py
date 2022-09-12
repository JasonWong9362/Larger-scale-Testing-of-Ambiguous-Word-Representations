import spacy
import numpy as np
from nltk.stem import WordNetLemmatizer


class ParamsControl(object):
    # preprocess params
    drop_type = "entire"  # "entire", "verb_only", "noun_only"
    win_size = [1, 2, 3, 4, 5, 100]  # -1

    # prasing params
    max_level = 2  # level 1 is root
    v_needed_deps = ["dobj", "nsubj", "nsubjpass"]
    n_needed_deps = ["dobj", "nsubj", "nsubjpass",
                     "amod", "nummod", "compound",
                     "prep", "pobj"]

    # cross validation
    cv_fold = 5

    # seed
    seed_num = 1000

    # grib serach params
    # ("add", "get_zero"),  # commutative
    # ("mult", "get_one"),  # commutative
    # ("phaser", "get_one"),
    # ("fuzz", "get_one"),
    # ("diag", "get_one")
    comp_list = [("phaser", "get_one")]  # commutative

    max_level_list = [10]  # < 10

    # "ignore", "as_input", "as_operator"
    deal_with_prep_list = ["ignore"]

    # "norm_1", "norm_2", "cos", "trace", "bures", "eigen_cos", "trace_sim", "trace_vec_cos"
    # "raw", "norm_1_point_wise", "hamming", "eigen_raw", "raw_dm_and_eigen",
    sim_metric_list = ["bures"]

    # "SVM", "LR", "MLP", "threshold"
    clf_list = ["threshold"]

    # "drop", "filter", "no_filter"
    tree_filter_list = ["no_filter"]

    # "parent_as_oper", "child_as_oper","noun_as_oper", "verb_as_oper"
    tree_direction_list = ["noun_as_oper"]

    # exp
    # e_factor_list = np.linspace(0, 2, num=21)
    e_factor_list = [0, 0.5, 1, 1.5, 2]
    # linear
    # e_factor_list = np.linspace(0, 0.5, num=50)

    wt_level = "word_level"  # "word_level"

    # is predict
    # if is_predict is True, there should be one set of params combination
    is_predict = False

    # visual
    print_wrong_predict_sample = False
    print_distant_distri = False

    # raw data path
    dataset_dir = "C:/Users/jeson/project/thesis/WiC_dataset"
    train_data_path = dataset_dir + "/train/train.data.txt"
    train_label_path = dataset_dir + "/train/train.gold.txt"
    val_data_path = dataset_dir + "/dev/dev.data.txt"
    val_label_path = dataset_dir + "/dev/dev.gold.txt"
    test_data_path = dataset_dir + "/test/test.data.txt"
    example_data_path = dataset_dir + "/train/example.txt"

    # tree path
    tree_dir = {"entire": "C:/Users/jeson/project/thesis/WiC_dataset/tree",  # drop_type: tree_dir
                "verb_only": "C:/Users/jeson/project/thesis/WiC_dataset/tree/verb_only",
                "noun_only": "C:/Users/jeson/project/thesis/WiC_dataset/tree/noun_only"}
    train_tree_path = tree_dir[drop_type] + "/train_tree.pkl"
    val_tree_path = tree_dir[drop_type] + "/val_tree.pkl"
    test_tree_path = tree_dir[drop_type] + "/test_tree.pkl"
    example_tree_path = tree_dir[drop_type] + "/example_tree.pkl"

    # pre-trained model
    model_file_path = "../../trained_dms/ms-word2dm-c5.model"
    spacy_model = spacy.load("en_core_web_trf")  # trf
    lem_model = WordNetLemmatizer()

    # test set prediction export path
    export_pred_file_path = "../../result/output.txt"
