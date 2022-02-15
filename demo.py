from river import evaluate, metrics, datasets, tree
from river.ensemble import BaggingClassifier
from IWE import IWE
from IWE_M import IWE_M

model2 = IWE(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
model3 = IWE_M(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)

models = [model2, model3]

for model in models:
    dataset = datasets.Elec2()
    eva = evaluate.progressive_val_score(dataset, model, metrics.GeometricMean())
    print(eva)


