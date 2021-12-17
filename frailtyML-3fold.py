import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
import time
from scipy import interp
import xgboost as xgb

random_state = np.random.RandomState(0)
#forest = RandomForestClassifier(random_state=random_state)
#forest = DecisionTreeClassifier(random_state=random_state)
forest = GradientBoostingClassifier(random_state=random_state)
#forest = HistGradientBoostingClassifier(random_state=random_state)
#forest = AdaBoostClassifier(random_state=random_state)

tprs = []
roc_aucs = []
pr_aucs = []
y_full = []
y_prob_full = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for i in range(1, 4):
    # load datasets
    X = pd.read_csv("train-x-fold"+str(i)+".csv", index_col='mergeid', low_memory=False)
    y = pd.read_csv("train-y-fold"+str(i)+".csv", index_col='mergeid', low_memory=False)
    X_test = pd.read_csv("test-x-fold"+str(i)+".csv", index_col='mergeid', low_memory=False)
    y_test_true = pd.read_csv("test-y-fold"+str(i)+".csv", index_col='mergeid', low_memory=False)
    
    # no skill classifier for comparison later
    dummy_model = DummyClassifier(strategy='stratified')
    dummy_model.fit(X, y)
    yhat = dummy_model.predict_proba(X_test)
    naive_probs = yhat[:, 1]
    # precision-recall auc for no-skill classifer
    ns_precision, ns_recall, _ = precision_recall_curve(y_test_true, naive_probs)
    dummy_auc_score = auc(ns_recall, ns_precision)
    print('No Skill PR AUC: %.4f' % dummy_auc_score)

    forest.fit(X, np.ravel(y))
    preds_test = forest.predict(X_test)

    # ROC AUC
    prediction = forest.predict_proba(X_test)
    prediction = prediction[:, 1]
    fpr, tpr, t = roc_curve(y_test_true, prediction, pos_label=1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)

    # plot ROC auc for each fold/loop
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
    
    # Precision-recall AUC
    precision, recall, thresholds = precision_recall_curve(y_test_true, prediction)
    pr_auc = auc(recall, precision)
    pr_aucs.append(pr_auc)
    y_full.append(y_test_true)
    y_prob_full.append(prediction)
    print("pr auc:" + str(pr_auc))
    print("fold:" + str(i) + "======================")
    
    # plot pr auc for each fold/loop
    plt.plot(recall, precision, marker='.', label='Fold %d AUC=%.4f' % (i, pr_auc))

    # other performance matrics
    #step1_rf_accuracy = accuracy_score(y_test_true, preds_test)
    #step1_cf_matrix = confusion_matrix(y_test_true, preds_test)
    #tn, fp, fn, tp = confusion_matrix(y_test_true, preds_test).ravel()
    #print(f'True Positives_test: {tp}')
    #print(f'False Positives_test: {fp}')
    #print(f'True Negatives_test: {tn}')
    #print(f'False Negatives_test: {fn}')
    step1_precision = precision_score(y_test_true, preds_test)
    step1_recall = recall_score(y_test_true, preds_test)
    #print("Accuracy_test: ",step1_rf_accuracy)
    #print("Confusion Matrix_test: ",step1_cf_matrix)
    #print("Precision score_test: ",step1_precision)
    #print("Recall score_test: ",step1_recall)
    #display = PrecisionRecallDisplay.from_predictions(y_test_true, prediction, name="Gradient Boosting")
    #_ = display.ax_.set_title("2-class Precision-Recall curve")

pr_std_auc = np.std(pr_aucs)
roc_std_auc = np.std(roc_aucs)
print("std. dev. pr auc: " + str(pr_std_auc))
print("std. dev. roc auc: " + str(roc_std_auc))

y_full = np.concatenate(y_full)
y_prob_full = np.concatenate(y_prob_full)
y_full_precision, y_full_recall, _ = precision_recall_curve(y_full, y_prob_full)
no_skill = len(y_full[y_full==1]) / len(y_full)

# plot pr curve
plt.plot(y_full_recall, y_full_precision, marker='.', color='black', label='Overall AUC=%0.4f' % (auc(y_full_recall, y_full_precision)))
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', color='gray')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for GBC with Imputed Walking Speed')
plt.legend()
plt.show()

# plot roc curve
'''
plt.plot([0,1],[0,1],linestyle = '--',lw = 1,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.4f)' % (mean_auc),lw=1, alpha=1)

std_auc = np.std(aucs)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ %0.4f std. dev." % (std_auc),
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic",
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for HGBC (baseline)')
plt.legend(loc="lower right")
plt.show()
'''