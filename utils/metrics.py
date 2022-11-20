
import numpy as np
from tqdm import tqdm

def compute_metrics(predictions, labels) :

    eval_recall = 0.0
    eval_ndcg = 0.0

    for pred, label in tqdm(zip(predictions, labels)) :
        recall = recallk(label, pred)
        ndcg = ndcgk(label, pred)

        eval_recall += recall
        eval_ndcg += ndcg

    eval_recall /= len(predictions)
    eval_ndcg /= len(predictions)        

    eval_score = 0.75*eval_recall + 0.25*eval_ndcg
    rets = {
        "recall" : eval_recall, 
        "ndcg" : eval_ndcg, 
        "score" : eval_score
    }

    return rets


def recallk(actual, predicted, k = 25):
    set_actual = set(actual)
    recall_k = len(set_actual & set(predicted[:k])) / min(k, len(set_actual))
    return recall_k


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def ndcgk(actual, predicted, k = 25):
    set_actual = set(actual)
    idcg = sum([1.0 / np.log(i + 2) for i in range(min(k, len(set_actual)))])
    dcg = 0.0
    unique_predicted = unique(predicted[:k])
    for i, r in enumerate(unique_predicted):
        if r in set_actual:
            dcg += 1.0 / np.log(i + 2)
    ndcg_k = dcg / idcg
    return ndcg_k
