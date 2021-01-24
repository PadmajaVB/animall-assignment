from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer


scorer = {'f2_score': make_scorer(fbeta_score, beta=2)}


def evaluating_metrics(y_val, y_val_pred, model_name):
    # In the above case since false negatives have higher penalty, we need to maximize recall value.
    # This can be best measured using F2 score

    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f2_score = fbeta_score(y_val, y_val_pred, beta=2.0)
    print('Result for %s: precision=%.3f, recall=%.3f, f2_score=%.3f without cross-validation' % (model_name, precision, recall, f2_score))


def cross_validation(estimator, X, y, scoring, cv, return_train_score=False):
    scores = cross_validate(estimator, X, y, scoring=scoring, cv=cv, return_train_score=return_train_score, n_jobs=-1)
    print('Average test f2 score in 10 fold cross validation = %.2f'%(scores['test_f2_score'].mean()))
    return scores


def random_forest(X_train, X_val, y_train, y_val):
    model_name = 'Random Forest'
    rf_model = BalancedRandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')

    rf_model.fit(X_train, y_train)
    y_val_pred = rf_model.predict(X_val)
    evaluating_metrics(y_val, y_val, model_name)
    return y_val_pred


def random_forest_with_cv(X, y):
    rf_model = BalancedRandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
    rf_model.fit(X, y)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cross_validation(rf_model, X, y, scoring=scorer, cv=cv)
    return rf_model
