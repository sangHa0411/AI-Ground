
import numpy as np
from tqdm import tqdm

"""
평가방법에 대한 의문점이 존재
1. 이미 봐왔던 것을 대상으로 추론해서 맞춘다?? -> 의미가 있는가???
"""
def compute_metrics(predictions, labels) :

    hr_1, hr_5, hr_10 = 0.0, 0.0, 0.0

    for i in tqdm(range(len(predictions))) :
        pred = predictions[i]
        label = labels[i]

        if label == pred[-1] :
            hr_1 += 1

        if label in pred[-5:] :
            hr_5 += 1

        if label in pred[-10:] :
            hr_10 += 1

    hr_1 /= len(predictions)
    hr_5 /= len(predictions)
    hr_10 /= len(predictions)

    return {'HR-1' : hr_1, 'HR-5' : hr_5, 'HR-10' : hr_10}