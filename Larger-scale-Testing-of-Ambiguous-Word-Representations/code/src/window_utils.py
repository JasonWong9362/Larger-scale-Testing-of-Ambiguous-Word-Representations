import numpy as np
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from density_matrices import DensityMatrices
from tqdm import tqdm
from params_control import *
from grid_search import *
from prasing_utils import *
from nltk.stem import WordNetLemmatizer
import string

"general import and export functions"


def read_raw_data(data_path, label_path, win_size, drop_type):
    # read sample from txt file
    df = pd.read_csv(data_path, delimiter="\t", header=None, dtype=str)

    # read label in train and val set
    if label_path != '':
        label_np = read_label(label_path)
        df['label'] = label_np
        # modify col rename
        df.columns = ['t_w', 'pos', 't_id', 'sen_1', 'sen_2', 'label']
    else:  # test set
        df.columns = ['t_w', 'pos', 't_id', 'sen_1', 'sen_2']

    # drop specific PoS type samples
    if drop_type == 'noun_only':
        df = df.drop(df[df.pos == 'V'].index)
    elif drop_type == 'verb_only':
        df = df.drop(df[df.pos == 'N'].index)

    # modify pos col
    df['pos'] = df.apply(lambda x: modify_pos(x.pos), axis=1)

    # drop punctuations in sentence
    df['sen_1'] = df.apply(lambda x: drop_punctuation(x.sen_1), axis=1)
    df['sen_2'] = df.apply(lambda x: drop_punctuation(x.sen_2), axis=1)

    # split target word index into two col
    df[['t_id_1', 't_id_2']] = df['t_id'].str.split('-', expand=True)

    # return sample and label
    if win_size == -1:
        return df

    else:
        # fetch a window of words and store in a array
        df['win_sen_1'] = df.apply(lambda x: get_win_words(x.sen_1, win_size, x.t_id_1), axis=1)
        df['win_sen_2'] = df.apply(lambda x: get_win_words(x.sen_2, win_size, x.t_id_2), axis=1)
        return df


def read_label(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    modify_lines = [line.replace('\n', '') for line in lines]
    labels = convert_label_to_bool(modify_lines)
    return np.array(labels)


def drop_punctuation(sen):
    punc_list = string.punctuation
    # punc_list = punc_list.replace('\'', '')
    # punc_list = punc_list.replace('-', '')
    punc_list = punc_list + '“' + '”' + '—' + '-' + '–'  # append other punctuation
    for character in punc_list:
        sen = sen.replace(character, '')
    return sen


def modify_pos(pos_type):
    if pos_type == 'V':
        return "VERB"
    elif pos_type == 'N':
        return "NOUN"


def convert_label_to_bool(label_lists):
    labels = []
    for label in label_lists:
        if label == 'T':
            labels.append(1)
        elif label == 'F':
            labels.append(0)
    return labels


def export_pred(target_path, pred):
    np.savetxt(target_path, pred, delimiter='\n', fmt='%1c')


def convert_label_to_str(pred_list):
    converted_pred = []
    for pred in pred_list:
        if pred == 0:
            converted_pred.append('F')
        else:
            converted_pred.append('T')
    return np.array(converted_pred)


"neighbour as context"


def get_win_words(sen, win_size, target_id):
    # lemmatize
    le = WordNetLemmatizer()
    # str2int
    target_id = int(target_id)
    words_id = list(range(target_id - win_size, target_id + win_size + 1))
    # remove target word
    # words_id.remove(target_id)
    excess_num = 0
    # remove punctuation
    sen = sen.translate(str.maketrans('', '', string.punctuation))

    # tokenize and lemmatize
    words = word_tokenize(sen)
    sen = [le.lemmatize(word) for word in words]
    sen = [word.lower() for word in sen]
    sen_len = len(sen)

    # if words_id excess
    # both side
    if len(words_id) > len(sen):
        # add space until the len of sen equal winSize
        # while len(words_id) > len(sen):
        #     sen.append('')
        return sen

    # left side
    while words_id[0] < 0:
        # move to right by adding 1 to all elements in the list
        words_id = [w_id + 1 for w_id in words_id]

    # right side
    while words_id[-1] > (sen_len - 1):
        words_id = [w_id - 1 for w_id in words_id]

    return [sen[w_id] for w_id in words_id]


def win_comp(df, dm_model, oov_get_wt, comp_func):
    # define variables for recording
    dms_1 = []
    dms_2 = []
    tw_entropy_list = []

    # iterate each sample
    for index, row in tqdm(df.iterrows(), leave=False):
        c1 = row["win_sen_1"]
        c2 = row["win_sen_2"]
        tw = row["t_w"]

        dm1 = comp_context(c1, dm_model, oov_get_wt, comp_func)
        dm2 = comp_context(c2, dm_model, oov_get_wt, comp_func)

        tw_entropy_list.append(get_entropy(dm_model.get_dm(tw, oov_get_wt)))

        dms_1.append(dm1)
        dms_2.append(dm2)

    return np.array(dms_1), np.array(dms_2), np.array(tw_entropy_list)


def comp_context(c_list, dm_model, oov_get_wt, comp_func):
    # get a zero or one matrix, according to comp func
    if comp_func == "add":
        comp_dm = np.zeros((17, 17))
        for w in c_list:
            comp_dm = comp_dm + dm_model.get_dm(w, oov_get_wt)
    elif comp_func == "mult":
        comp_dm = np.ones((17, 17))
        for w in c_list:
            comp_dm = comp_dm * dm_model.get_dm(w, oov_get_wt)

    comp_dm = comp_dm / comp_dm.trace()
    return comp_dm


# experiment 1 grid search
if __name__ == "__main__":

    dm_model = DensityMatrices.load(ParamsControl.model_file_path)

    # combination of params
    combination = product(ParamsControl.comp_list, ParamsControl.win_size)
    gs_params = [gs_param for gs_param in combination]

    # define report dataframe
    report = pd.DataFrame()

    # grid search
    for gs_param in gs_params:
        print(str(gs_param))

        # define params dict
        params = {"win_size": gs_param[1],
                  "drop_type": ParamsControl.drop_type,
                  "spacy_model": ParamsControl.spacy_model,
                  "lem_model": ParamsControl.lem_model,
                  "dm_model": DensityMatrices.load(ParamsControl.model_file_path),
                  "comp": gs_param[0][0],
                  "oov_get_wt": gs_param[0][1],
                  "model": ParamsControl.wt_level}

        # load raw load, neighbours as context
        train_df = read_raw_data(ParamsControl.train_data_path, ParamsControl.train_label_path, params["win_size"],
                                 params["drop_type"])
        val_df = read_raw_data(ParamsControl.val_data_path, ParamsControl.val_label_path, params["win_size"],
                               params["drop_type"])

        "compose"
        train_dms_1, train_dms_2, train_elist = win_comp(train_df, dm_model, params["oov_get_wt"], params["comp"])
        val_dms_1, val_dms_2, val_elist = win_comp(val_df, dm_model, params["oov_get_wt"], params["comp"])

        "concat train and val set labels, word lists and entropy lists"
        y = np.concatenate((train_df["label"].values, val_df["label"].values), axis=0)
        e_list = np.concatenate((train_elist, val_elist))
        print(np.count_nonzero(e_list == 0))
        e_list = np.where(e_list < 0, 0, e_list)
        e_list = e_list.reshape((-1, 1))
        # plt.hist(e_list, bins=1000)
        # plt.show()

        for metric in ParamsControl.sim_metric_list:
            # print(sim_metric)

            # train and val sim
            X_sim = get_similarity(train_dms_1, train_dms_2, val_dms_1, val_dms_2, metric)
            # convert Nan to zero
            X_sim = np.nan_to_num(X_sim)

            # get test set similarity
            if ParamsControl.is_predict:
                X_sim_test = get_similarity(test_dms_1, test_dms_2, 0, 0, metric)
                # convert Nan to zero
                X_sim_test = np.nan_to_num(X_sim_test)
                test_elist = np.array(test_elist)
                test_elist = test_elist.reshape((-1, 1))
                modified_X_sim = modified_sim(X_sim_test, test_elist, ParamsControl.e_factor_list[0], 1, metric)
                # convert Nan to zero
                modified_X_sim = np.nan_to_num(modified_X_sim)

            for clf_name in ParamsControl.clf_list:
                # print(clf_name)

                for e_factor in ParamsControl.e_factor_list:
                    # linear
                    # modified_e_list = (-e_factor)*(e_list-x)+1
                    # exp
                    modified_X_sim = modified_sim(X_sim, e_list, e_factor, 1, metric)
                    # convert Nan to zero
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
