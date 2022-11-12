
import numpy as np
from tqdm import tqdm

def compute_metrics(predictions, labels) :

    hr_1, hr_5, hr_10 = 0.0, 0.0, 0.0

    for i in tqdm(range(len(predictions))) :
        pred_args = np.argsort(predictions[i])

        label = labels[i]

        if label == pred_args[-1] :
            hr_1 += 1

        if label in pred_args[-5:] :
            hr_5 += 1

        if label in pred_args[-10:] :
            hr_10 += 1

    hr_1 /= len(predictions)
    hr_5 /= len(predictions)
    hr_10 /= len(predictions)

    return {'HR-1' : hr_1, 'HR-5' : hr_5, 'HR-10' : hr_10}