from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.neighbors import KNeighborsClassifier

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "rf": ensemble.RandomForestClassifier(),
    "svm": svm.SVC(probability=True),
    "knn": KNeighborsClassifier(),
    "extra_trees": ensemble.ExtraTreesClassifier(),
    "gradient_boosting": ensemble.GradientBoostingClassifier(),
    "adaboost": ensemble.AdaBoostClassifier(),
    "xgboost": xgb.XGBClassifier(n_estimators=1000,max_depth=6,learning_rate=0.03,objective="binary:logistic",eval_metric="auc",use_label_encoder=False,early_stopping_rounds=50),
    "lightgbm": lgb.LGBMClassifier(),
    "catboost": cb.CatBoostClassifier(verbose=0),
    "logistic_regression": linear_model.LogisticRegression(),
}