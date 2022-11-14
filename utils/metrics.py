
import numpy as np
from tqdm import tqdm


TOPK = [10, 15, 25]

def compute_metrics(predictions, labels) :

    recalls = [0.0 for _ in range(len(TOPK))]

    for i in tqdm(range(len(predictions))) :
        label = labels[i]

        for j in range(len(TOPK)) :
            k = TOPK[j]
            pred = predictions[i][:k]
            recall_k = len(set(label) & set(pred)) / len(set(label))

            recalls[j] += recall_k    

    recalls = [r/len(predictions) for r in recalls]        
    recalls = {f'recall_{k}' : recalls[i] for i, k in enumerate(TOPK)}
    return recalls