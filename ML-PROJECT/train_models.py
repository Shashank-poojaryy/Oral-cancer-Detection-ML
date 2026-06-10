import numpy as np
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (GridSearchCV,
                                     RandomizedSearchCV,
                                     StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)
import sys

sys.stdout.reconfigure(encoding='utf-8')

PROC    = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\processed")
MODELS  = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\outputs\models")
OUTPUTS = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\outputs")

OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# LOAD DATA
X_train = np.load(PROC / 'X_train.npy')
X_test  = np.load(PROC / 'X_test.npy')
y_train = np.load(PROC / 'y_train.npy')
y_test  = np.load(PROC / 'y_test.npy')

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"y_train: {Counter(y_train)} | y_test: {Counter(y_test)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# HELPER FUNCTION
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)  * 100
    prec = precision_score(y_test, y_pred,
                           pos_label=1,
                           zero_division=0) * 100
    rec  = recall_score(y_test, y_pred,
                        pos_label=1,
                        zero_division=0)   * 100
    f1   = f1_score(y_test, y_pred,
                    pos_label=1,
                    zero_division=0)       * 100
    print(f"""
─────────────────────────────────────────────
  Model     : {name}
─────────────────────────────────────────────
  Accuracy  : {acc:.2f}%
  Precision : {prec:.2f}%
  Recall    : {rec:.2f}%  ← Cancer detection
  F1 Score  : {f1:.2f}%
─────────────────────────────────────────────""")
    return {'Model': name,
            'Accuracy': round(acc,2),
            'Precision': round(prec,2),
            'Recall': round(rec,2),
            'F1_Score': round(f1,2)}

results = []

# MODEL 1 — LOGISTIC REGRESSION
print("\n>>> Training Model 1: Logistic Regression...")
lr = LogisticRegression(C=1.0,
                         max_iter=2000,
                         solver='lbfgs',
                         class_weight='balanced',
                         random_state=42)
lr.fit(X_train, y_train)

cv_scores = cross_val_score(lr, X_train, y_train,
                             cv=cv, scoring='recall')
print(f"  CV Recall: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

joblib.dump(lr, MODELS / 'logistic_regression.pkl')
results.append(evaluate_model('Logistic Regression', lr, X_test, y_test))


# MODEL 2 — SVM (GridSearch Tuned)
print("\n>>> Training Model 2: SVM with GridSearchCV...")
svm_params = {
    'C':     [1, 10, 100],
    'gamma': ['scale', 'auto']
}
svm_base = SVC(kernel='rbf',
               class_weight='balanced',
               probability=True,
               random_state=42)
svm_grid = GridSearchCV(svm_base,
                         svm_params,
                         cv=cv,
                         scoring='recall',
                         n_jobs=-1,
                         verbose=0)
svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_

print(f"  Best SVM params : {svm_grid.best_params_}")
print(f"  Best CV Recall  : {svm_grid.best_score_*100:.2f}%")

joblib.dump(best_svm, MODELS / 'svm.pkl')
results.append(evaluate_model('SVM (Tuned)', best_svm, X_test, y_test))


# MODEL 3 — RANDOM FOREST (Tuned)
print("\n>>> Training Model 3: Random Forest...")
rf_params = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [10, 15, 20, None],
    'min_samples_split':[2, 5]
}
rf_base = RandomForestClassifier(class_weight='balanced',
                                  random_state=42,
                                  n_jobs=-1)
rf_search = RandomizedSearchCV(rf_base,
                                rf_params,
                                n_iter=10,
                                cv=cv,
                                scoring='recall',
                                random_state=42,
                                n_jobs=-1,
                                verbose=0)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

print(f"  Best RF params  : {rf_search.best_params_}")
print(f"  Best CV Recall  : {rf_search.best_score_*100:.2f}%")

joblib.dump(best_rf, MODELS / 'random_forest.pkl')
results.append(evaluate_model('Random Forest (Tuned)', best_rf, X_test, y_test))


# MODEL 4 — DECISION TREE
print("\n>>> Training Model 4: Decision Tree...")
dt = DecisionTreeClassifier(max_depth=10,
                              class_weight='balanced',
                              criterion='gini',
                              random_state=42)
dt.fit(X_train, y_train)

cv_scores = cross_val_score(dt, X_train, y_train,
                             cv=cv, scoring='recall')
print(f"  CV Recall: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

joblib.dump(dt, MODELS / 'decision_tree.pkl')
results.append(evaluate_model('Decision Tree', dt, X_test, y_test))


# MODEL 5 — KNN (Auto Best-K)
print("\n>>> Training Model 5: KNN (finding best k)...")
best_k = 3
best_k_score = 0

for k in [3, 5, 7, 9, 11]:
    knn_temp = KNeighborsClassifier(
                   n_neighbors=k,
                   metric='euclidean',
                   weights='distance')
    scores = cross_val_score(knn_temp,
                              X_train, y_train,
                              cv=cv,
                              scoring='recall')
    print(f"  k={k} → CV Recall: {scores.mean()*100:.2f}%")
    if scores.mean() > best_k_score:
        best_k_score = scores.mean()
        best_k = k

print(f"  Best k selected : {best_k}")
best_knn = KNeighborsClassifier(
               n_neighbors=best_k,
               metric='euclidean',
               weights='distance')
best_knn.fit(X_train, y_train)
joblib.dump(best_knn, MODELS / 'knn.pkl')
results.append(evaluate_model(f'KNN (k={best_k})', best_knn, X_test, y_test))


# MODEL 6 — NAIVE BAYES
print("\n>>> Training Model 6: Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)

cv_scores = cross_val_score(nb, X_train, y_train,
                             cv=cv, scoring='recall')
print(f"  CV Recall: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

joblib.dump(nb, MODELS / 'naive_bayes.pkl')
results.append(evaluate_model('Naive Bayes', nb, X_test, y_test))


# MODEL 7 — SOFT VOTING ENSEMBLE
print("\n>>> Training Model 7: Soft Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('svm', best_svm),
        ('rf',  best_rf),
        ('lr',  lr)
    ],
    voting='soft',
    n_jobs=-1
)
ensemble.fit(X_train, y_train)

cv_scores = cross_val_score(ensemble,
                             X_train, y_train,
                             cv=cv,
                             scoring='recall')
print(f"  CV Recall: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

joblib.dump(ensemble, MODELS / 'voting_ensemble.pkl')
results.append(evaluate_model('Voting Ensemble', ensemble, X_test, y_test))


# FINAL COMPARISON TABLE
df = pd.DataFrame(results)
df = df.sort_values('Recall', ascending=False)
df = df.reset_index(drop=True)
df.index += 1

print("\n")
print("═"*65)
print("        FINAL MODEL COMPARISON — Sorted by Cancer Recall")
print("═"*65)
print(f"{'#':<4}{'Model':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
print("─"*65)
for i, row in df.iterrows():
    print(f"{i:<4}{row['Model']:<28}{row['Accuracy']:>9.2f}%{row['Precision']:>10.2f}%{row['Recall']:>8.2f}%{row['F1_Score']:>8.2f}%")
print("═"*65)

df.to_csv(OUTPUTS / 'model_comparison.csv', index=False)
print(f"\n✓ Comparison table saved → outputs/model_comparison.csv")

best_row = df.iloc[0]
print(f"""
🏆 Best Model   : {best_row['Model']}
   Accuracy     : {best_row['Accuracy']}%
   Precision    : {best_row['Precision']}%
   Recall       : {best_row['Recall']}%
   F1 Score     : {best_row['F1_Score']}%

Meaning: {best_row['Recall']/10:.1f} out of 10 cancer cases correctly identified by this model.
""")
