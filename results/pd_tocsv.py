import pandas as pd

results_tensorflow = pd.DataFrame({ "Modelo":["Modelo 1", "Modelo 2","Modelo 3", "Modelo 4","Modelo 5"],
    "Tempo (s)":[19.12,84.04,4.60,5.56,12.70],
    "Loss":[0.2020,0.0955,0.4060,0.0016,0.2188],
    "Accuracy":[0.9572,0.9704,1.0000,1.0000,0.9637],
    "Val. Loss":[0.5530,0.5903,0.4408,0.9070,0.7273],
    "Val. Accuracy":[0.8667,0.8875,0.9550,0.7866,0.7451]
})

results_tfbert = pd.DataFrame({ "Modelo":["Modelo 1", "Modelo 2"],
    "Tempo (s)":[2082.58,1540.92],
    "Loss":[0.4403,0.0908],
    "Accuracy":[0.8022,0.9603],
    "Val. Loss":[0.0728,0.5093],
    "Val. Accuracy":[0.9878,0.8875]
})

results_tensorflow.to_csv("results_tensorflow.csv",index=False,encoding='utf8')
results_tfbert.to_csv("results_tfbert.csv",index=False,encoding='utf8')