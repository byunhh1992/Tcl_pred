import numpy as np


preds = np.load("./results/preds.npy")
trues = np.load("./results/trues.npy")

ercentage_errors = np.zeros_like(preds)
percentage_errors = np.abs((preds - trues))

mpe = np.mean(percentage_errors)
print(mpe)

debug = True