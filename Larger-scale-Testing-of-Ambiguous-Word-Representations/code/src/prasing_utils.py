import spacy
import math
from anytree import Node, PostOrderIter, RenderTree
from spacy import displacy
from window_utils import *
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import os
from collections import Counter
import matplotlib.pyplot as plt
from similarity import *

""" load raw data, build tree for sentence """


# integrate load and compose function
def load_and_compose(data_path, label_path, tree_path, param_dict):
    # load data
    dataset = load_tree(data_path=data_path,
                        label_path=label_path,
                        tree_path=tree_path,
                        win_size=param_dict["win_size"],
                        drop_type=param_dict["drop_type"],
                        spacy_model=param_dict["spacy_model"],
                        lem_model=param_dict["lem_model"])

    # example tree (verify)
    # example_tree = dataset.iat[100, 0]  # tree 1, 100th example

    # compose
    dms_1, dms_2, w_list_1, w_list_2, oov, e_list = compose(dm_model=param_dict["dm_model"],
                                                            comp=param_dict["comp"],
                                                            dataset=dataset,
                                                            max_level=param_dict["max_level"],
                                                            oov_get_wt=param_dict["oov_get_wt"],
                                                            v_needed_deps=param_dict["v_needed_deps"],
                                                            n_needed_deps=param_dict["n_needed_deps"],
                                                            deal_with_prep=param_dict["deal_with_prep"],
                                                            tree_filter=param_dict["tree_filter"],
                                                            tree_direction=param_dict["tree_direction"],
                                                            model=param_dict["model"])

    # load label
    if label_path != "":
        y = dataset["label"].values
        return dms_1, dms_2, w_list_1, w_list_2, y, e_list
    else:
        return dms_1, dms_2, w_list_1, w_list_2, e_list


# load or build tree and label, if build tree, include read raw data function
def load_tree(data_path, label_path, tree_path, win_size, drop_type, spacy_model, lem_model):
    if os.path.exists(tree_path):
        # read tree
        tree_df = pd.read_pickle(tree_path)

        # check_same_target_word(tree_df)

    else:  # tree path not exist
        # load and preprocess sentence
        dataset_df = read_raw_data(data_path, label_path, win_size, drop_type)

        # parsing and get related words in a tree form
        tree_df = encode_sen_to_tree(dataset_df, spacy_model, lem_model)

        # verify if the target of two trees are the same
        check_same_target_word(tree_df)

        if label_path != "":
            tree_df["label"] = dataset_df["label"]

        # save df
        tree_df.to_pickle(tree_path)

    return tree_df


# convert sentences into trees using parsing function
def encode_sen_to_tree(df, spacy_model, lem_model):
    # progress bar for df
    tqdm.pandas()

    # build tree for two sentences
    df["dep_tree_1"] = df.progress_apply(lambda x: parsing(x.sen_1, x.t_w, x.t_id_1, x.pos, spacy_model, lem_model),
                                         axis=1)
    df["dep_tree_2"] = df.progress_apply(lambda x: parsing(x.sen_2, x.t_w, x.t_id_2, x.pos, spacy_model, lem_model),
                                         axis=1)

    check_empty_tree(df[["dep_tree_1", "dep_tree_2"]])

    return df[["dep_tree_1", "dep_tree_2"]]


# preprocessed steps before building a tree
def parsing(sen, t_w, t_id, t_pos, spacy_model, lem_model):
    t_id = int(t_id)

    # creating Doc object for sen
    doc = spacy_model(sen)

    # extract dependency dictionary from Doc obj
    dep_dict = displacy.parse_deps(doc)
    deps = dep_dict["arcs"]

    # the doc object and dependency dictionary can be un-equal length
    t_id, trusted_w_list = deal_with_no_equal_len(sen, t_w, t_id, doc, dep_dict, lem_model)

    # modify dependency
    modify_deps = [ordered_dep(dep) for dep in deps]

    # construct tree base on dependency
    try:
        dep_tree = build_dep_tree(Node(t_id, w=trusted_w_list[t_id]["word"], pos=t_pos, dm=0, sen=sen),
                                  trusted_w_list, modify_deps)
    except:
        print("fail to build tree. sen:" + str(sen) + " t_w:" + str(t_w))

    return dep_tree


# check if the length of Doc obj equal to the dependency dictionary, and deal with un-equal length sample
def deal_with_no_equal_len(sen, t_w, true_id, doc, dep_dict, lem_model):
    # define target id variable
    t_id = None
    # define trusted word list
    trusted_w_list = []
    # define length equal variable
    is_length_equal = True
    finded = False

    for i, word in enumerate(doc):
        tmp_w = word.lemma_.lower()  # lemma and lower case
        tmp_w_2 = lem_model.lemmatize(tmp_w)
        tmp_w_3 = word.lower_
        tmp_w_4 = lem_model.lemmatize(tmp_w_3)
        tmp_w = tmp_w.replace(" ", "")  # remove space
        # define element in trusted_w_list
        tmp_dict = {"word": tmp_w, "pos": word.pos_}
        # find target word id
        if not finded:
            if true_id - 2 <= i <= true_id + 2:
                if t_w == tmp_w or t_w == tmp_w_2 or t_w == tmp_w_3 or t_w == tmp_w_4:
                    t_id = i
                    finded = True
        # append
        trusted_w_list.append(tmp_dict)

    if not finded:
        for i, word in enumerate(doc):
            tmp_w_3 = word.lower_
            if match(t_w, tmp_w_3, true_id, i):
                t_id = i
                break

    if not finded:
        for i, word in enumerate(doc):
            tmp_w_3 = word.lower_
            if match2(t_w, tmp_w_3, true_id, i):
                t_id = i
                break

    if not finded:
        for i, word in enumerate(doc):
            tmp_w_3 = word.lower_
            if match3(t_w, tmp_w_3, true_id, i):
                t_id = i
                break

    # return only root node if do not find target word
    if t_id is None:
        print("not find target word id.")
        print("sen:" + str(sen) + " target word:" + str(t_w) + " is_length_equal" + str(is_length_equal))
        return Node(-1, w="", pos="", dm=0, sen=sen)

    return t_id, trusted_w_list


# return modified-order dependency dictionary
def ordered_dep(dic):
    # the return tuple represent a dependency between 2 words, it is order (first element point at the second)
    if dic["dir"] == "left":
        return {"start": dic["end"], "end": dic["start"], "type": dic["label"]}
    else:  # "dir" == "right"
        return {"start": dic["start"], "end": dic["end"], "type": dic["label"]}


# build tree recursively given a sentence
def build_dep_tree(parent, w_list, deps):
    order = 1

    # no deps
    if check_dep_exist(parent, deps):
        return parent

    # exist dep
    else:
        for i, dep in enumerate(deps):
            node_attr_direction = ""
            # dep direction "out" of the root node

            if dep["start"] == parent.name or dep["end"] == parent.name:

                if dep["start"] == parent.name:
                    node_attr_direction = "out"
                    child_id = dep["end"]
                # dep direction "in" of the root node
                elif dep["end"] == parent.name:
                    node_attr_direction = "in"
                    child_id = dep["start"]

                child = Node(child_id, w=w_list[child_id]["word"], pos=w_list[child_id]["pos"],
                             dep_type=dep["type"], dep_direction=node_attr_direction, dm=0, parent=parent,
                             order=order)
                # drop cur node for recursive
                child_deps = deps[:i] + deps[i + 1:]
                build_dep_tree(child, w_list, child_deps)

                order += 1

            else:
                continue

        return parent


# check if there is a dependency of a node(word) exist, so that we can continue build tree
def check_dep_exist(cur_node, deps):
    not_exist = True
    for dep in deps:
        if dep["start"] == cur_node.name:
            not_exist = False
            break
        if dep["end"] == cur_node.name:
            not_exist = False
            break
    return not_exist


# check empty tree in dataframe
def check_empty_tree(trees_df):
    for index, row in trees_df.iterrows():
        if type(row["dep_tree_1"]) != Node:
            print("emtry tree 1, index:" + str(index))
        if type(row["dep_tree_1"]) != Node:
            print("emtry tree 1, index:" + str(index))
    return 0


def check_same_target_word(trees_df):
    for index, row in trees_df.iterrows():
        if row["dep_tree_1"].w != row["dep_tree_2"].w:
            print("tree1 tw:" + row["dep_tree_1"].w + " sen:" + str(row["dep_tree_1"].sen))
            print("tree2 tw:" + row["dep_tree_2"].w + " sen:" + str(row["dep_tree_2"].sen))
    return 0


def match(t_w, tmp_w_3, t_id, i):
    if t_id - 1 <= i <= t_id + 1:
        if len(tmp_w_3) > int(len(t_w) * 0.5 + 1):
            if t_w[:int(min(len(t_w), len(tmp_w_3) * 0.5 + 1))] == tmp_w_3[:int(min(len(t_w), len(tmp_w_3) * 0.5 + 1))]:
                return True
    return False


def match2(t_w, tmp_w_3, t_id, i):
    if t_id - 1 <= i <= t_id + 1:
        if len(tmp_w_3) > int(len(t_w) * 0.5):
            if t_w[:int(min(len(t_w), len(tmp_w_3) * 0.5))] == tmp_w_3[:int(min(len(t_w), len(tmp_w_3) * 0.5))]:
                return True
    return False


def match3(t_w, tmp_w_3, t_id, i):
    if t_id - 1 <= i <= t_id + 1:
        if t_w[-2:] == tmp_w_3[-2:]:
            return True
    return False


"""compose"""


# iterate two columns of trees for composing and further statistic
def compose(dm_model, comp, dataset, max_level, oov_get_wt, v_needed_deps, n_needed_deps, deal_with_prep,
            tree_filter, tree_direction, model):
    # predefine variavle
    vocab = dm_model.vocab.itos  # for OOV

    # define variables for recording
    dms_1 = []
    dms_2 = []
    tw_entropy_list = []
    t1_w_list_list = []
    t2_w_list_list = []
    dep_type_set_noun = []
    dep_type_set_verb = []
    oov_list = []
    sum_same_sim = 0

    # iterate each sample
    for index, row in tqdm(dataset.iterrows(), leave=False):

        tree_1 = row["dep_tree_1"]
        tree_2 = row["dep_tree_2"]

        dm1 = np.zeros((17, 17))
        dm2 = np.zeros((17, 17))
        t1_w_list = []
        t2_w_list = []

        dm1, oov1, t1_w_list, t1_dep_type, t1_pos = compose_tree(dm_model, vocab, comp, tree_1, max_level,
                                                                 oov_get_wt, v_needed_deps, n_needed_deps,
                                                                 deal_with_prep, tree_filter, tree_direction, model)
        dm2, oov2, t2_w_list, t2_dep_type, t2_pos = compose_tree(dm_model, vocab, comp, tree_2, max_level,
                                                                 oov_get_wt, v_needed_deps, n_needed_deps,
                                                                 deal_with_prep, tree_filter, tree_direction, model)

        # check whether the two dms are the same
        # if check_dms_not_equal(dm1, dm2) == 0:
        #     print("Sen1:" + str(tree_1.sen) + "   " + "neighbour:" + str(t1_w_list))
        #     print("Sen2:" + str(tree_2.sen) + "   " + "neighbour:" + str(t2_w_list))

        # check eigen error
        try:
            _, _ = np.linalg.eigh(dm1)
            _, _ = np.linalg.eigh(dm2)
        except:
            print("check eigen error")
            print("Sen1:" + str(tree_1.sen) + "   " + "neighbour:" + str(t1_w_list))
            print("Sen2:" + str(tree_2.sen) + "   " + "neighbour:" + str(t2_w_list))

        # store
        dms_1.append(dm1)
        dms_2.append(dm2)
        t1_w_list_list.append(t1_w_list)
        t2_w_list_list.append(t2_w_list)
        oov_list.extend(oov1)
        oov_list.extend(oov2)
        tw_entropy_list.append(get_entropy(dm_model.get_dm(tree_1.w, oov_get_wt)))

    # tree stat
    #     if t1_pos == "VERB":
    #         dep_type_set_verb.extend(t1_dep_type)
    #         dep_type_set_verb.extend(t2_dep_type)
    #     elif t1_pos == "NOUN":
    #         dep_type_set_noun.extend(t1_dep_type)
    #         dep_type_set_noun.extend(t2_dep_type)
    #
    # # dict record dep type
    # dict_dep_noun = Counter(dep_type_set_noun)
    # dict_dep_verb = Counter(dep_type_set_verb)
    #
    # # sort dict by value
    # dict_dep_noun = {k: v for k, v in sorted(dict_dep_noun.items(), key=lambda item: item[1])}
    # dict_dep_verb = {k: v for k, v in sorted(dict_dep_verb.items(), key=lambda item: item[1])}
    # print(dict_dep_noun.keys())
    # print(len(dict_dep_noun.keys()))
    # print(dict_dep_verb.keys())
    # print(len(dict_dep_verb.keys()))
    #
    # # visual dep type
    # f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    # ax1.bar(dict_dep_noun.keys(), dict_dep_noun.values(), color="g")
    # ax1.axhline(y=len(dms_1) * 0.05, color="r", linestyle="-")
    # ax2.bar(dict_dep_verb.keys(), dict_dep_verb.values(), color="g")
    # ax2.axhline(y=len(dms_1) * 0.05, color="r", linestyle="-")
    # plt.show()

    return np.array(dms_1), np.array(dms_2), t1_w_list_list, t2_w_list_list, oov_list, np.array(tw_entropy_list)


# composing given a tree
def compose_tree(dm_model, vocab, comps, root, max_level, oov_get_wt, v_needed_deps, n_needed_deps, deal_with_prep,
                 tree_filter, tree_direction, model):
    # target word PoS
    t_pos = root.pos

    confirm_list, w_list, dep_type = modify_tree(root, tree_filter, max_level, t_pos, v_needed_deps, n_needed_deps)

    oov_list = []
    if len(confirm_list) == 1:  # only one node
        return dm_model.get_dm(root.w, oov_get_wt), oov_list, w_list, dep_type, t_pos

    # loop deps
    for node in confirm_list:
        if node.is_root:
            # check root node dm has int type
            if type(root.dm) == int:
                print(node.sen)
                # print(confirm_list)
            break
        else:
            # check OOV
            if node.w not in vocab:
                if node.w not in oov_list:
                    oov_list.append(node.w)
            if node.parent.w not in vocab:
                if node.parent.w not in oov_list:
                    oov_list.append(node.parent.w)

            # compose child and parent, child be the operator and parent be the target
            try:
                if model == "word_level":
                    node.root.dm = dm_model.compose(node, node.root, comps, oov_get_wt, deal_with_prep, tree_direction)
                elif model == "sen_level":
                    node.parent.dm = dm_model.compose(node, node.parent, comps, oov_get_wt, deal_with_prep, tree_direction)
            except:
                print("compose error")

    return root.dm, oov_list, w_list, dep_type, t_pos


"modify tree"


def modify_tree(root, tree_filter, max_level, t_pos, v_needed_deps, n_needed_deps):
    if tree_filter == "drop":
        drop_deps(root)

    # PostOrderIter
    post_order = [node for node in PostOrderIter(root, maxlevel=max_level)]

    # dep type
    dep_type = []
    for node in post_order:
        if hasattr(node, "dep_type"):
            dep_type.append(node.dep_type + "_" + node.dep_direction)

    # no filter
    if tree_filter == "no_filter" or tree_filter == "drop":
        confirm_list = post_order

    # filter
    if tree_filter == "filter":
        confirm_list, _ = drop_deps_cut_tree(t_pos, v_needed_deps, n_needed_deps, post_order)

    # modify node order of post order list
    modify_compose_order(confirm_list)

    w_list = [node.w for node in confirm_list]

    # print tree
    # for pre, fill, node in RenderTree(root):
    #     print("%s%s (%s)" % (pre, node.w, node.pos))

    return confirm_list, w_list, dep_type


def modify_compose_order(post_order_list):
    root = post_order_list[-1]
    height = root.height
    same_parent = []
    child_group = []
    # iterate the level of tree
    for h in range(1, height + 1):
        for node_id, node in enumerate(post_order_list):
            if node.depth > h:
                child_group.append(node)
            elif node.depth == h:
                child_group.append(node)
                same_parent.append(child_group)
                child_group = []
            elif node.depth < h:
                if len(same_parent) > 1:
                    # sort
                    sorted_same_parent = same_parent_sort(same_parent)
                    # flatten
                    sorted_same_parent = [item for sublist in sorted_same_parent for item in sublist]
                    # append
                    end_id = node_id
                    front_id = end_id - len(sorted_same_parent)
                    post_order_list[front_id: end_id] = sorted_same_parent
                child_group = []
                same_parent = []
        child_group = []
        same_parent = []

    return post_order_list


# increasing
def same_parent_sort(same_parent_list):
    for iter_num in range(len(same_parent_list) - 1, 0, -1):
        for idx in range(iter_num):
            prior_1 = get_prior(same_parent_list[idx][-1].dep_type)
            prior_2 = get_prior(same_parent_list[idx+1][-1].dep_type)
            # prior_1 = same_parent_list[idx][-1].name
            # prior_2 = same_parent_list[idx + 1][-1].name

            if prior_1 < prior_2:
                # swag
                same_parent_list[idx], same_parent_list[idx + 1] = same_parent_list[idx + 1], same_parent_list[idx]
    return same_parent_list


# the lower the priority is, the more preceding it is
def get_prior(dep_type):
    prior_dict = {"dobj": 0}

    if dep_type in list(prior_dict.keys()):
        return_prior = prior_dict[dep_type]
    else:
        return_prior = 100
    return return_prior


# given a tree iteration, drop node with corresponding relations and check validation
def drop_deps_cut_tree(t_pos, v_needed_deps, n_needed_deps, post_order):
    comp_list = []
    if t_pos == "VERB":
        pos_needed_deps = v_needed_deps
        # only verb, compound
        post_order = deal_with_compound(post_order)
    elif t_pos == "NOUN":
        pos_needed_deps = n_needed_deps
    else:
        print("pos not found error.")

    for i in range(0, len(post_order) - 1):  # last element is root, no type
        if post_order[i].dep_type in pos_needed_deps:
            comp_list.append(post_order[i])
    comp_list.append(post_order[-1])

    # check if the parent of a node is in the list
    confirm_list = []
    for node in comp_list:
        if node.is_root:
            confirm_list.append(node)
        elif ancestors_in_list(node.ancestors, comp_list):
            confirm_list.append(node)
    w_list = [node.w for node in confirm_list]

    return confirm_list, w_list


# correct the error of spacy prasing model, only verb
def deal_with_compound(post_order):
    if len(post_order) == 2:  # and post_order[0].dep_type == 'compound':
        post_order[0].dep_direction = 'in'
        post_order[0].dep_type = 'dobj'
        post_order[-1].pos = 'VERB'
        return post_order
    else:
        return post_order


# check if ancestors nodes in the given list
def ancestors_in_list(ancestors, node_list):
    is_in_list = True
    for ancestor in ancestors:
        if ancestor not in node_list:
            is_in_list = False
            break
    return is_in_list


# drop the node that is preposition
def drop_deps(root):
    node_list = [node for node in PostOrderIter(root)]
    confirmed_list = []
    for node in node_list:
        if not node.is_root:  # or node.pos == "ADP" or node.w == 'of'
            if node.pos == "DET" or node.pos == "PART" or node.pos == "AUX" \
                    or node.pos == "SYM" or node.pos == "X" or node.pos == "SPACE":
                for child in node.children:
                    child.parent = node.parent
                node.parent = None
                node.children = tuple()
            else:
                confirmed_list.append(node)
        else:
            confirmed_list.append(node)
    return confirmed_list, [node.w for node in confirmed_list]


"utils"


# compute entropy given a dm
def get_entropy(dm):
    eigvals, _ = np.linalg.eigh(dm)
    entropy = 0
    for eigval in eigvals:
        if eigval > 0:
            entropy += eigval * (math.log(eigval, math.e))
    return -entropy


def print_sample(node, w_list):
    print("target word: " + str(node.w))
    print("sen: " + str(node.sen))
    print("w_list: " + str(w_list))
    print("--------------------------------------------------")


# check if two given density matrix is identical
def check_dms_equal(dm1, dm2):
    return (dm1 == dm2).all()


if __name__ == "__main__":
    i = 0
