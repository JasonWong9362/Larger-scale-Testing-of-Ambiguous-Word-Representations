import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.linalg import sqrtm
import numpy.linalg as la
import math

"""get_similarity"""


def get_similarity(train_dms_1, train_dms_2, val_dms_1, val_dms_2, metric):
    if type(val_dms_1) == int and type(val_dms_2) == int:
        dms_1 = train_dms_1
        dms_2 = train_dms_2
    else:
        # concat
        dms_1 = np.concatenate((train_dms_1, val_dms_1), axis=0)
        dms_2 = np.concatenate((train_dms_2, val_dms_2), axis=0)

    # split into two flatten dm
    if metric.startswith("norm") or metric.startswith("cos") or metric.startswith("hamming"):
        dms_1 = flatten_dm(dms_1)
        dms_2 = flatten_dm(dms_2)

    if metric == "norm_1":
        result = norm_1(dms_1, dms_2)

    elif metric == "norm_2":
        result = norm_2(dms_1, dms_2)

    elif metric == "cos":
        result = cos(dms_1, dms_2)

    elif metric == "norm_1_point_wise":
        result = norm_1_point_wise(dms_1, dms_2)

    elif metric == "hamming":
        result = hamming(dms_1, dms_2)

    elif metric == "trace":
        result = trace(dms_1, dms_2)

    elif metric == "bures":
        result = bures(dms_1, dms_2)

    elif metric == "raw":
        f_dms1, f_dms2 = flatten_dm(dms_1), flatten_dm(dms_2)
        result = np.concatenate((f_dms1, f_dms2), axis=1)

    elif metric == "eigen_raw":
        result = get_eigen(dms_1, dms_2, "raw")

    elif metric == "eigen_cos":
        result = get_eigen(dms_1, dms_2, "cos")
        result = result.reshape((-1, 1))

    elif metric == "raw_dm_and_eigen":
        # eigen
        result_1 = get_eigen(dms_1, dms_2, "raw")

        # raw
        f_dms1, f_dms2 = flatten_dm(dms_1), flatten_dm(dms_2)
        result_2 = np.concatenate((f_dms1, f_dms2), axis=1)

        result = np.concatenate((result_1, result_2), axis=1)

    elif metric == "raw_dm_eigen_diff":
        # eigen
        result_1 = get_eigen(dms_1, dms_2, "diff")

        # raw
        f_dms1, f_dms2 = flatten_dm(dms_1), flatten_dm(dms_2)
        result_2 = np.abs(f_dms1 - f_dms2)

        result = np.concatenate((result_1, result_2), axis=1)

    elif metric == "trace_sim":
        result = trace_sim(dms_1, dms_2)

    elif metric == "trace_vec_cos":
        result = trace_vec_cos(dms_1, dms_2)

    # # plot distri
    # plt.hist(result, bins=1000)
    # plt.show()

    return result


def flatten_dm(dms):
    flatten_dms = []
    for dm in dms:
        if type(dm) == int:
            flatten_dms.append(np.zeros(17 * 17))
            print(dms)
            print('dm has a int type')
        else:
            flatten_dms.append(dm.flatten())
    return np.array(flatten_dms)


def norm_1(dms_1, dms_2):
    result = np.sum(np.abs(dms_1 - dms_2), axis=1)
    return result.reshape((-1, 1))


def norm_2(dms_1, dms_2):
    result = np.sqrt(np.sum(np.square(dms_1 - dms_2), axis=1))
    return result.reshape((-1, 1))


def cos(dms_1, dms_2):
    dot = np.sum(dms_1 * dms_2, axis=1)
    norm = np.linalg.norm(dms_1, axis=1) * np.linalg.norm(dms_2, axis=1)
    result = np.divide(dot, norm, out=np.zeros_like(dot), where=norm != 0)
    return result.reshape((-1, 1))


def norm_1_point_wise(dms_1, dms_2):
    return np.abs(dms_1 - dms_2)


def hamming(dms_1, dms_2):
    return np.round(dms_1, 3) == np.round(dms_2, 3)


def trace(dms_1, dms_2):
    distance = []
    for dm_id in range(dms_1.shape[0]):
        distance.append(tracedist_rho_rho(dms_1[dm_id], dms_2[dm_id]))
    distance = np.array(distance)
    return distance.reshape((-1, 1))


def trace_sim(dms_1, dms_2):
    distance = np.trace(np.matmul(dms_1, dms_2), axis1=1, axis2=2)  # matmul
    distance = 1 - distance
    return distance.reshape((-1, 1))


def trace_vec_cos(dms_1, dms_2):
    distance = []
    for dm_id in range(dms_1.shape[0]):
        dm_1_diag = dms_1[dm_id].diagonal()
        dm_2_diag = dms_2[dm_id].diagonal()
        cosine = np.dot(dm_1_diag, dm_2_diag) / (np.linalg.norm(dm_1_diag) * np.linalg.norm(dm_2_diag))
        distance.append(cosine)
    distance = np.array(distance)
    return distance.reshape(-1, 1)


# def trace_sim(dms_1, dms_2):
#     distance = []
#     for dm_id in range(dms_1.shape[0]):
#         eval_1, evec_1 = np.linalg.eigh(dms_1[dm_id])
#         eval_2, evec_2 = np.linalg.eigh(dms_2[dm_id])
#         modify_dm_1 = np.zeros((17, 17))
#         modify_dm_2 = np.zeros((17, 17))
#         for e_id in range(1):
#             tmp_evec_1 = evec_1[:, -(e_id+1)].reshape((1, -1))
#             tmp_evec_2 = evec_2[:, -(e_id+1)].reshape((1, -1))
#             modify_dm_1 += eval_1[-(e_id+1)]*(tmp_evec_1.T@tmp_evec_1)
#             modify_dm_2 += eval_2[-(e_id+1)]*(tmp_evec_2.T@tmp_evec_2)
#         distance.append(np.trace(np.matmul(modify_dm_1, modify_dm_2)))
#     distance = np.array(distance)
#     return distance.reshape((-1, 1))


def get_dm_sqrt(dm):
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals[eigvals <= 0] = 0
    dm_sqrt = (eigvecs * np.sqrt(
        eigvals)) @ eigvecs.T  # element-wise multiplication in bracket achieves matrix multiplication with diagonal matrix
    return dm_sqrt


def bures(dms_1, dms_2):
    distance = []
    for dm_id in range(dms_1.shape[0]):
        distance.append(infidelity_rho_rho(dms_1[dm_id], dms_2[dm_id]))
    distance = np.array(distance)
    return distance.reshape((-1, 1))


def get_eigen(dms_1, dms_2, mode):  # element-wise norm1, hamming, cos
    result = []
    for i in range(dms_1.shape[0]):
        evalues1, evectors1 = np.linalg.eigh(dms_1[i])
        evalues2, evectors2 = np.linalg.eigh(dms_2[i])

        if mode == "raw":
            needed_vector_1 = evectors1[:, -5:]
            needed_vector_1 = needed_vector_1.flatten()
            needed_val_1 = evalues1[-5:]
            needed_vector_2 = evectors2[:, -5:]
            needed_vector_2 = needed_vector_2.flatten()
            needed_val_2 = evalues2[-5:]
            result.append(np.concatenate((needed_vector_1, needed_val_1, needed_vector_2, needed_val_2)))

        elif mode == "cos":
            eval1_set = evalues1[-5:]
            eval2_set = evalues2[-5:]
            evec1_set = evectors1[:, -5:]
            evec2_set = evectors2[:, -5:]

            # cos sim
            # dot = np.abs(np.sum(needed_vector_1 * needed_vector_2, axis=0))
            # result.append(dot)

            dot = switch_vector(evec1_set.T, evec2_set.T, eval1_set, eval2_set)
            result.append(dot)

        elif mode == "diff":
            needed_vector_1 = evectors1[:, -5:]
            needed_vector_1 = needed_vector_1.flatten()
            needed_val_1 = evalues1[-5:]
            needed_vector_2 = evectors2[:, -5:]
            needed_vector_2 = needed_vector_2.flatten()
            needed_val_2 = evalues2[-5:]
            vec_diff = np.abs(needed_vector_1 - needed_vector_2)
            val_diff = np.abs(needed_val_1 - needed_val_2)
            result.append(np.concatenate((vec_diff, val_diff)))

    return np.array(result)


def switch_vector(evec1_set, evec2_set, eval1_set, eval2_set):
    cos_sum = 0
    sum_count = 0
    for i, evec1 in enumerate(evec1_set):
        for j, evec2 in enumerate(evec2_set):
            if abs(np.dot(evec1, evec2)) > 0.9:
                cos_sum += eval1_set[i] * eval2_set[j]
                sum_count += 1
                break

    if sum_count != 5:
        print("error!, sum not enough.")
    norm = np.linalg.norm(eval1_set) * np.linalg.norm(eval2_set)
    return cos_sum / norm


def tracedist_rho_rho(rho, sigma):
    """
    Calculate the trace distance :math:`D = 0.5 Tr | rho - sigma |` with
    :math:`|A| = \sqrt{A^{\dagger} A}`.

    **Arguments**

    rho : 2d numpy array
        First density matrix.

    sigma : 2d numpy array
        Second density matrix.
    """
    tmp = rho - sigma
    return 0.5 * np.sum(la.svd(tmp, compute_uv=False))


def infidelity_rho_rho(rho, sigma):
    """
    Calculate the infidelity :math:`I = 1 - \sqrt{\sqrt{rho} sigma \sqrt{rho}}`

    **Arguments**

    rho : 2d numpy array
        First density matrix.

    sigma : 2d numpy array
        Second density matrix.
    """
    tmp = get_dm_sqrt(rho)
    tmp = np.dot(tmp, np.dot(sigma, tmp))
    infidelity = np.real(1 - np.trace(get_dm_sqrt(tmp)))
    return infidelity


""" get classifier """


class mlp_clf:
    def __init__(self, input_shape):
        model = Sequential()
        model.add(Dense(100, input_shape=input_shape, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        self.model = model

    def fit(self, X, y, X_val, y_val):
        model = self.model

        # define callback
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", mode='auto', factor=0.5, patience=5, verbose=0)
        es = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=0, min_delta=0.005)

        model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, validation_data=(X_val, y_val), epochs=200, batch_size=32, verbose=0,
                  callbacks_list=[reduce_lr, es])

    def predict(self, X):
        pred = self.model.predict(X[:1])
        return pred


class threshold_clf:
    def __init__(self, granularity=100):
        self.granularity = granularity
        self.pred_threshold = 0
        self.reversed = False

    def fit(self, X, y):
        threshold_result = []
        threshold_reversed_result = []
        # max_val = np.amax(X, axis=0)
        # min_val = np.amin(X, axis=0)
        max_val = 0
        min_val = 2
        # searching range
        search_range = np.linspace(min_val, max_val, self.granularity, endpoint=False)
        # searching threshold
        for threshold in search_range:
            pred_y = np.array([X > threshold])
            pred_y = pred_y.reshape(pred_y.shape[1])

            reversed_pred_y = np.array([X < threshold])
            reversed_pred_y = reversed_pred_y.reshape(reversed_pred_y.shape[1])

            threshold_result.append(np.sum(pred_y == y))
            threshold_reversed_result.append(np.sum(reversed_pred_y == y))

        threshold_result = np.array(threshold_result)
        threshold_reversed_result = np.array(threshold_reversed_result)

        if np.max(threshold_reversed_result) > np.max(threshold_result):
            self.pred_threshold = search_range[np.argmax(threshold_reversed_result)]
            self.reversed = True

        else:
            self.pred_threshold = search_range[np.argmax(threshold_result)]

    def predict(self, X):
        if self.reversed:
            pred_y = np.array([X < self.pred_threshold])
        else:
            pred_y = np.array([X > self.pred_threshold])
        return pred_y.reshape(pred_y.shape[1])


def get_clf(clf_name):
    if clf_name == "LR":
        clf = LogisticRegression()

    elif clf_name == "SVM":
        clf = svm.SVC()

    elif clf_name == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=0,
                            max_iter=200, early_stopping=True)

    elif clf_name == "threshold":
        clf = threshold_clf()

    return clf


""" get classification result"""


def get_classification_result(y_true, y_pred):
    acc = round(accuracy_score(y_true, y_pred), 3)
    # prec = round(precision_score(y_true, y_pred), 3)
    # recall = round(recall_score(y_true, y_pred), 3)
    # f1 = round(f1_score(y_true, y_pred), 3)
    return acc  # , prec, recall, f1


def get_dm_sqrt(dm):
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals[eigvals <= 0] = 0
    dm_sqrt = (eigvecs * np.sqrt(
        eigvals)) @ eigvecs.T  # element-wise multiplication in bracket achieves matrix multiplication with diagonal matrix
    return dm_sqrt


if __name__ == "__main__":
    print(np.multiply([1, 2], [1, 2]))

