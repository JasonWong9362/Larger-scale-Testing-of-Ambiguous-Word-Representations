from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prasing_utils import *
from sklearn.model_selection import KFold
from nltk.stem import WordNetLemmatizer
from similarity import *
from window_utils import *
from params_control import *


def plot_distant_distriX_sim(X_sim):
    fig, axs = plt.subplots(3, 1, tight_layout=True)
    zeros = X_sim[np.where(y == 0)]
    ones = X_sim[np.where(y == 1)]
    source = [X_sim, zeros, ones]
    for i, ax in enumerate(axs):
        ax.hist(source[i], bins=1000)
        ax.set_xlim(np.min(X_sim), np.max(X_sim))
        ax.set_ylim(0, 100)
    plt.show()


def modified_sim(sim, e_list, e_factor, x_offset, metric_):  # x_offset could be the mean or median of the e-distribution
    modified_e_list = np.power(e_list + (x_offset - 1), e_factor)
    if metric_ == "trace_sim":
        modified_sim = np.divide(sim, modified_e_list, out=np.ones_like(sim), where=modified_e_list != 0)
    else:
        modified_sim = np.multiply(sim, modified_e_list, out=np.zeros_like(sim), where=modified_e_list != 0)
    return modified_sim


if __name__ == "__main__":

    # combination of params
    combination = product(ParamsControl.comp_list, ParamsControl.max_level_list,
                          ParamsControl.deal_with_prep_list,
                          ParamsControl.tree_filter_list, ParamsControl.tree_direction_list)
    gs_params = [gs_param for gs_param in combination]

    # define report dataframe
    report = pd.DataFrame()

    # grid search
    for gs_param in gs_params:
        print(str(gs_param))

        # define params dict
        params = {"win_size": ParamsControl.win_size,
                  "drop_type": ParamsControl.drop_type,
                  "spacy_model": ParamsControl.spacy_model,
                  "lem_model": ParamsControl.lem_model,
                  "dm_model": DensityMatrices.load(ParamsControl.model_file_path),
                  "comp": gs_param[0][0],
                  "max_level": gs_param[1],
                  "oov_get_wt": gs_param[0][1],
                  "deal_with_prep": gs_param[2],
                  "v_needed_deps": ParamsControl.v_needed_deps,
                  "n_needed_deps": ParamsControl.n_needed_deps,
                  "tree_filter": gs_param[3],
                  "tree_direction": gs_param[4],
                  "model": ParamsControl.wt_level}

        # example tree
        # example_dm_1, example_dm_2 = load_and_compose(ParamsControl.example_data_path,
        #                                               '',
        #                                               ParamsControl.example_tree_path,
        #                                               params)

        # get tree
        train_set = load_tree(data_path=ParamsControl.train_data_path,
                              label_path=ParamsControl.train_label_path,
                              tree_path=ParamsControl.train_tree_path,
                              win_size=params["win_size"],
                              drop_type=params["drop_type"],
                              spacy_model=params["spacy_model"],
                              lem_model=params["lem_model"])
        val_set = load_tree(data_path=ParamsControl.val_data_path,
                            label_path=ParamsControl.val_label_path,
                            tree_path=ParamsControl.val_tree_path,
                            win_size=params["win_size"],
                            drop_type=params["drop_type"],
                            spacy_model=params["spacy_model"],
                            lem_model=params["lem_model"])
        concat_set = pd.concat([train_set, val_set])

        # sample tree given id
        # sample = train_set.iat[300, 0]

        # example_dms_1, example_dms_2, example_w_list_1, example_w_list_2, example_y = load_and_compose(
        #     ParamsControl.example_data_path,
        #     "",
        #     ParamsControl.example_tree_path,
        #     params)

        # # train and val sim
        # example_X_sim = get_similarity(example_dms_1, example_dms_2, 0, 0, "eigen_cos")

        "load tree and compose"
        train_dms_1, train_dms_2, train_w_list_1, train_w_list_2, train_y, train_elist = load_and_compose(
            ParamsControl.train_data_path,
            ParamsControl.train_label_path,
            ParamsControl.train_tree_path,
            params)

        val_dms_1, val_dms_2, val_w_list_1, val_w_list_2, val_y, val_elist = load_and_compose(
            ParamsControl.val_data_path,
            ParamsControl.val_label_path,
            ParamsControl.val_tree_path,
            params)

        if ParamsControl.is_predict:
            test_dms_1, test_dms_2, test_w_list_1, test_w_list_2, test_elist = load_and_compose(
                ParamsControl.test_data_path,
                "",
                ParamsControl.test_tree_path,
                params)

        "concat train and val set labels, word lists and entropy lists"
        y = np.concatenate((train_y, val_y), axis=0)
        w_list_1 = np.concatenate((train_w_list_1, val_w_list_1), axis=0)
        w_list_2 = np.concatenate((train_w_list_2, val_w_list_2), axis=0)
        e_list = np.concatenate((train_elist, val_elist))
        e_list = np.where(e_list < 0, 0, e_list)
        e_mean = np.mean(e_list)
        e_median = np.median(e_list)
        e_list = e_list.reshape((-1, 1))

        # matplotlib.rcParams['font.family'] = 'Times New Roman'
        # matplotlib.rcParams['font.size'] = '16'
        # plt.hist(e_list, bins=1000)
        # plt.xlabel('Von Neumann entropy')
        # plt.ylabel('Number of words')
        # plt.show()

        for metric in ParamsControl.sim_metric_list:
            # print(sim_metric)

            # train and val sim
            X_sim = get_similarity(train_dms_1, train_dms_2, val_dms_1, val_dms_2, metric)
            # convert Nan to zero
            X_sim = np.nan_to_num(X_sim)
            # plt.hist(X_sim, bins=1000)
            # plt.show()

            # get test set similarity
            if ParamsControl.is_predict:
                X_sim_test = get_similarity(test_dms_1, test_dms_2, 0, 0, metric)

                # convert Nan to zero
                X_sim_test = np.nan_to_num(X_sim_test)
                test_elist = test_elist.reshape((-1, 1))

                # def modified_sim(sim, e_list, e_factor, x_offset):

                modified_X_sim_test = modified_sim(X_sim_test, test_elist, ParamsControl.e_factor_list[0], 1, metric)
                modified_X_sim_test = np.nan_to_num(modified_X_sim_test)

            for clf_name in ParamsControl.clf_list:
                # print(clf_name)

                for e_factor in ParamsControl.e_factor_list:
                    modified_X_sim = modified_sim(X_sim, e_list, e_factor, 1, metric)
                    modified_X_sim = np.nan_to_num(modified_X_sim)

                    for i in range(modified_X_sim.shape[0]):
                        if modified_X_sim[i] > 10:
                            modified_X_sim[i] = 0

                    if ParamsControl.print_distant_distri:
                        # plot distant distribution
                        plot_distant_distriX_sim(modified_X_sim)

                    # record result
                    train_result = []
                    val_result = []

                    if not ParamsControl.is_predict:

                        # cross validation
                        n_kf = 1
                        kf = KFold(n_splits=ParamsControl.cv_fold, shuffle=True,
                                   random_state=ParamsControl.seed_num)
                        for train_index, test_index in kf.split(modified_X_sim, y):
                            X_train, X_test = modified_X_sim[train_index], modified_X_sim[test_index]
                            y_train, y_test = y[train_index], y[test_index]

                            clf = get_clf(clf_name)
                            clf.fit(X_train, y_train)
                            pred_train = clf.predict(X_train)
                            pred_val = clf.predict(X_test)

                            # record
                            acc_train = get_classification_result(y_train, pred_train)
                            acc_val = get_classification_result(y_test, pred_val)
                            train_result.append(acc_train)
                            val_result.append(acc_val)

                            n_kf += 1

                        len_inequality = 0
                        number_of_wrong = 0

                        if ParamsControl.print_wrong_predict_sample:
                            # print wrong-predicted samples
                            pred = clf.predict(modified_X_sim)
                            id_list = np.where(y != pred)
                            for i in id_list[0]:
                                print("target word: " + concat_set.iat[i, 0].w)
                                print(str(concat_set.iat[i, 0].sen) + " word list:" + str(w_list_1[i]))
                                print(concat_set.iat[i, 1].sen + " word list:" + str(w_list_2[i]))
                                print("label:" + str(y[i]))
                                print("tw entropy: " + str(e_list[i]))
                                print("measure: " + str(X_sim[i]))
                                print("modify measure: " + str(modified_X_sim[i]))
                                print("======================================")
                                number_of_wrong += 1
                                if len(w_list_1[i]) != len(w_list_2[i]):
                                    len_inequality += 1

                        print("num of sample:" + str(len(modified_X_sim)))
                        print("len_inequality:" + str(len_inequality) + "/" + str(number_of_wrong))

                    # for test
                    if ParamsControl.is_predict:
                        clf = get_clf(clf_name)
                        clf.fit(modified_X_sim, y)
                        # export test set prediction
                        pred_test = clf.predict(modified_X_sim_test)
                        export_pred(ParamsControl.export_pred_file_path, convert_label_to_str(pred_test))

                    train_result = np.array(train_result)
                    val_result = np.array(val_result)

                    # define a record
                    record_dict = {"comp": gs_param[0][0],
                                   "max_level": gs_param[1],
                                   "oov_get_wt": gs_param[0][1],
                                   "deal_with_prep": gs_param[2],
                                   "tree_filter": gs_param[3],
                                   "tree_direction": gs_param[4],
                                   "sim_metric": metric,
                                   "clf": clf,
                                   "e_factor": e_factor,
                                   "train_avg": round(np.mean(train_result), 3) * 100,
                                   "train_std": round(np.std(train_result), 3) * 100,
                                   "val_avg": round(np.mean(val_result), 3) * 100,
                                   "val_std": round(np.std(val_result), 3) * 100}
                    # append record
                    report = report.append(record_dict, ignore_index=True)

    # save result
    report.to_csv("../../result/report.csv")
