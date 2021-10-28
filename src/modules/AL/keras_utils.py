import numpy as np

import sys

############## Import modules ##############
sys.path.append("../../")

from modules.utils import compute_classification_report


def test_keras_ensemble(sess, models, testX, testY, save_path):
    
    weights = [1.0/len(models) for _ in range(len(models))]

    outputs = []

    for model in models:
        output = model.predict(testX)
        output = output[:testY.shape[0]]
        outputs.append(output)

    outputs = np.array(outputs)

    outputs = np.tensordot(outputs, weights, axes=((0), (0)))
    predicted = np.argmax(outputs, axis=1)
    
    report = compute_classification_report(testY, predicted, verbose=1, save=save_path, store_dict=True)