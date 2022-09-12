import numpy as np
import math
import io
import pickle


class DensityMatrices(object):

    def __init__(self, vocab, dm):
        """
        vocab: torchtext FIELD.vocab object
        dm: numpy array containing density matrices, shape (vocab_size, dim, dim)
        """
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.dm = dm
        self.dim = dm.shape[1]

    def normalise(self):
        # Normalise to unit trace
        self.dm = self.dm / np.reshape(self.dm.trace(axis1=1, axis2=2), (self.dm.shape[0], 1, 1))

    def contains(self, word):
        return word in self.vocab.stoi

    def get_dm(self, word, oov_get_wt):
        if word not in self.vocab.itos:
            if word.isnumeric():
                word_index = self.vocab.stoi["one"]
                return self.dm[word_index]
            else:
                if oov_get_wt == "get_zero":
                    return np.zeros((self.dim, self.dim))
                elif oov_get_wt == "get_one":
                    return np.ones((self.dim, self.dim))
                else:
                    print("oov_get_wt set error.")
                    return 0
        else:
            word_index = self.vocab.stoi[word]
            return self.dm[word_index]

    def similarity(self, word1, word2):
        word1_dm = self.get_dm(word1)
        word2_dm = self.get_dm(word2)
        # Efficient way to compute trace of matrix product
        trace = (word1_dm * word2_dm.T).sum().item()
        # Normalise
        trace = trace / (math.sqrt((word1_dm ** 2).sum()) * math.sqrt((word2_dm ** 2).sum()))
        return trace

    def sequence_similarity(self, sequence1, sequence2, methods, pos_tags):
        sims = {}
        for method in methods:
            dm1 = self.compose(sequence1, method, pos_tags)
            dm2 = self.compose(sequence2, method, pos_tags)
            if dm1 is None or dm2 is None:
                sims[method] = None
            else:
                trace = (dm1 * dm2.T).sum().item()
                normalised_inner_product = trace / (math.sqrt((dm1 ** 2).sum()) * math.sqrt((dm2 ** 2).sum()))
                sims[method] = normalised_inner_product
        return sims, dm1, dm2

    def compose(self, node, parent_node, method, oov_get_wt, deal_with_prep, tree_direction):

        # check if dm is exist
        if type(node.dm) is int:
            node.dm = self.get_dm(node.w, oov_get_wt)
        if type(parent_node.dm) is int:
            parent_node.dm = self.get_dm(parent_node.w, oov_get_wt)

        if method == "mult":
            result = node.dm * parent_node.dm

        elif method == "add":
            result = node.dm + parent_node.dm

        elif method == "fuzz":
            result = self.fuzz_phaser("fuzz", method[method.find("_") + 1:], node.dm, parent_node.dm,
                                      node.pos, parent_node.pos, deal_with_prep, parent_node.is_root, tree_direction)

        elif method == "phaser":
            result = self.fuzz_phaser("phazer", method[method.find("_") + 1:], node.dm, parent_node.dm,
                                      node.pos, parent_node.pos, deal_with_prep, parent_node.is_root, tree_direction)

        elif method == "diag":
            result = self.diag(node.dm, parent_node.dm, node.pos, parent_node.pos, deal_with_prep)

        elif method == "fuzz_max_entropy":
            result, entropy = self.fuzz_max_entropy(node.dm, parent_node.dm)
            return result, round(entropy, 3)

        else:
            print("invalid comp method:" + method)

        # Normalise composed matrix
        if not np.all((result == 0)):  # zero matrix no need norm
            result = result / np.trace(result)

        return result

    def save(self, file_path):
        with open(file_path, "wb") as model_file:
            pickle.dump(self, model_file, protocol=4)

    @staticmethod
    def load(file_path):
        # Opening a file in ‘rb’ mode means that the file is opened for reading (r) in binary (b) mode
        return renamed_load(open(file_path, "rb"))

    """Composition methods"""

    def diag(self, w0_dm, w1_dm, w0_pos, w1_pos, deal_with_prep):

        diag_0 = np.diag(w0_dm)
        diag_1 = np.diag(w1_dm)
        # reshape for matmult
        diag_0 = diag_0.reshape((-1, 1))
        diag_1 = diag_1.reshape((-1, 1))

        # compute both direction
        d1_dm = np.matmul(diag_1, diag_0.T)
        d2_dm = np.matmul(diag_0, diag_1.T)

        # select the dm with lower entropy
        return self.check_dm_entropy(d1_dm, d2_dm)

        # previous method, check PoS to determine direction

        # if w0_pos == "VERB":
        #     result = np.matmul(diag_1, diag_0.T)
        # elif w1_pos == "VERB":
        #     result = np.matmul(diag_0, diag_1.T)
        # else:
        #     print("w0_pos:" + str(w0_pos) + "   w1_pos:" + str(w1_pos))
        #     result = np.matmul(diag_0, diag_1.T)
        # return result

    def fuzz_phaser(self, composer, operator, w0_dm, w1_dm, w0_pos, w1_pos, deal_with_prep, w1_is_root, tree_direction):

        # kmult or bmult
        if composer == "fuzz":
            compose_func = self.kmult
        else:
            compose_func = self.bmult

        if tree_direction == "parent_as_oper":
            result = compose_func(w1_dm, w0_dm)

        elif tree_direction == "child_as_oper":
            result = compose_func(w0_dm, w1_dm)

        elif tree_direction == "noun_as_oper":
            if w0_pos == "NOUN":  # or w0_pos == "NUM" or w0_pos == "PRON" or w0_pos == "PROPN"
                result = compose_func(operator_dm=w0_dm, input_dm=w1_dm)
            elif w1_pos == "NOUN":  # or w1_pos == "NUM" or w1_pos == "PRON" or w1_pos == "PROPN"
                result = compose_func(operator_dm=w1_dm, input_dm=w0_dm)
            else:
                # print("w0_pos:" + str(w0_pos) + "   w1_pos:" + str(w1_pos))
                result = compose_func(operator_dm=w1_dm, input_dm=w0_dm)

        elif tree_direction == "verb_as_oper":
            if w0_pos == "VERB":
                result = compose_func(operator_dm=w0_dm, input_dm=w1_dm)
            elif w1_pos == "VERB":
                result = compose_func(operator_dm=w1_dm, input_dm=w0_dm)
            else:
                # print("w0_pos:" + str(w0_pos) + "   w1_pos:" + str(w1_pos))
                result = compose_func(operator_dm=w1_dm, input_dm=w0_dm)

        elif tree_direction == "entropy":
            d1_dm = compose_func(w0_dm, w1_dm)
            d2_dm = compose_func(w1_dm, w0_dm)
            result = self.check_dm_entropy(d1_dm, d2_dm)

        return result

    def fuzz_max_entropy(self, w0_dm, w1_dm):
        result = self.kmult(w1_dm, w0_dm)
        entropy = self.get_entropy(result)
        return result, entropy

    @staticmethod
    def bmult(operator_dm, input_dm):
        operator_dm_sqrt = DensityMatrices.get_dm_sqrt(operator_dm)  # ugly OOP?
        # inv = np.linalg.pinv(operator_dm_sqrt)
        # result = inv @ operator_dm_sqrt @ input_dm @ operator_dm_sqrt @ inv  # matmul
        result = operator_dm_sqrt @ input_dm @ operator_dm_sqrt
        return result

    @staticmethod
    def get_dm_sqrt(dm):
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals[eigvals <= 0] = 0
        dm_sqrt = (eigvecs * np.sqrt(
            eigvals)) @ eigvecs.T  # element-wise multiplication in bracket achieves matrix multiplication with diagonal matrix
        return dm_sqrt

    @staticmethod
    def kmult(operator_dm, input_dm):
        eigvals, eigvecs = np.linalg.eigh(operator_dm)
        eigvals[eigvals <= 0] = 0
        result = np.zeros(shape=operator_dm.shape)
        for i in range(12, operator_dm.shape[0]):
            if eigvals[i] > 0:
                eigvec_outer = np.outer(eigvecs.T[i, :], eigvecs.T[i, :])
                result += eigvals[i] * (eigvec_outer @ input_dm @ eigvec_outer)
        return result

    @staticmethod
    def get_entropy(dm):
        eigvals, _ = np.linalg.eigh(dm)
        entropy = 0
        for eigval in eigvals:
            if eigval > 0:
                entropy += eigval * (math.log(eigval, math.e))
        return -entropy

    def check_dm_entropy(self, dm1, dm2):
        entropy_1 = self.get_entropy(dm1)
        entropy_2 = self.get_entropy(dm2)

        if entropy_1 > entropy_2:
            return dm2
        else:
            return dm1


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "word2dm":
            renamed_module = "density_matrices"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


if __name__ == "__main__":
    dm_model = DensityMatrices.load("../../trained_dms/ms-word2dm-c5.model")
    word_dm = dm_model.get_dm("he", "get_zero")
    eigvals, eigvecs = np.linalg.eigh(word_dm)

    A = np.array([[3, 1]])
    B = np.array([[49, 2]])
    print(A.T @ B)
    # print(DensityMatrices.fuzz(A, B))
