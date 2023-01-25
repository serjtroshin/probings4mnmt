import numpy as np
def loading(path, is_label=False):
    output = np.load(path, allow_pickle=True)
    processed = []
    for vector in output:
        processed.append(list(vector))
    if is_label:
        processed = np.reshape(processed, (-1, 1))
    else:
        processed = np.reshape(processed, (-1, 768))
    return processed

