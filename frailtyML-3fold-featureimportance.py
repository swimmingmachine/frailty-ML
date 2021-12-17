import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

random_state = np.random.RandomState(0)
#forest = RandomForestClassifier(random_state=random_state)
#forest = DecisionTreeClassifier(random_state=random_state)
forest = GradientBoostingClassifier(random_state=random_state)
#forest = HistGradientBoostingClassifier(random_state=random_state)
#forest = xgb.XGBClassifier(objective="binary:logistic", use_label_encoder=False)

for i in range(1, 4):
    # load datasets
    X = pd.read_csv("imputed-train-x-fold"+str(i)+".csv", index_col='mergeid', low_memory=False)
    y = pd.read_csv("train-y-fold"+str(i)+".csv", index_col='mergeid', low_memory=False)
    X_test = pd.read_csv("imputed-test-x-fold"+str(i)+".csv", index_col='mergeid', low_memory=False)
    y_test_true = pd.read_csv("test-y-fold"+str(i)+".csv", index_col='mergeid', low_memory=False)
    
    forest.fit(X, np.ravel(y))

    result = permutation_importance(
        forest, X, y, n_repeats=10, random_state=42, n_jobs=2
        )
    sorted_idx = result.importances_mean.argsort()
    
    # plot feature importance chart for each fold
    fig, ax = plt.subplots()
    ax.boxplot(
        result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx]
        )
    ax.set_title('Feature Importances in Step 2 (Fold %d' % i + ' training set)')
    fig.tight_layout()

plt.legend(loc="lower right")
plt.show()

