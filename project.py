import pandas as pd
import numpy as np
import argparse
from collections import Counter
from utils import divide_artists, clean_artists, sum_counters, get_models, evaluate_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, RobustScaler, QuantileTransformer,\
    MinMaxScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
import lightgbm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--pre_processing', type=int, default=1, help='Type of pre-processing pipeline, 1 or 2')
parser.add_argument('--model', type=str, default='lightgbm', help='Model selected')
parser.add_argument('--graphs',default=False, action='store_true', required=False, help='Create graphs for the report')
parser.add_argument('--tuning_robust_scaler', default=False, action='store_true', required=False, help='tuning of robust scaler range')
args = parser.parse_args()

# read data
df = pd.read_csv("data/dev.tsv", sep="\t")


# -------------------- Preprocessing pipelines --------------------
if args.pre_processing == 1:
    df["artists"] = clean_artists(df["artists"])

    # value_counts() for "stringed" artists
    print("Top 10 artists")
    pd.value_counts(df["artists"])[:10]

elif args.pre_processing == 2:
    df["artists"] = divide_artists(df["artists"])

    counter_per_rows = [] 

    for row in df["artists"].values:
        counter_per_rows.append((Counter(row)))

    # value_counts of "listed" artists
    print("Top 10 artists")
    global_counter = sum_counters(counter_per_rows)
    global_counter.most_common(10)

# drop the id column, since it's unique for each row
df.drop(columns=["id"], inplace=True)

# we can divide our set into train and validation set
X_train, X_valid, y_train, y_valid = train_test_split(df.drop(["mode"], axis=1),df["mode"], random_state=42, stratify=df["mode"])

# distinguish between numerical columns from the non-numerical ones
num_cols = [col for col in list(X_train.columns) if col not in ["artists"]]
art_cols = "artists"

if args.pre_processing == 1:
    # 1st preprocessing pipeline (TF-DF on artists)
    transformer = ColumnTransformer([
            #('minmax', MinMaxScaler(), num_cols),
            #('rob', RobustScaler(), rob_cols),
            ('stand', StandardScaler(), num_cols),
            ('artists_tfid', TfidfVectorizer(stop_words="english", use_idf=False, binary=True), art_cols)
        ], 
        remainder='drop'
    )
elif args.pre_processing == 2:
    # 2nd Preprocessing pipeline (MLB on artists)
    # Customed MultiLabelBinarizer that takes as input 3 elements.
    class MyMultiLabelBinarizer(TransformerMixin):
        def __init__(self, *args, **kwargs):
            self.encoder = MultiLabelBinarizer(*args, **kwargs)
            self.classes_ = self.encoder.classes
        def fit(self, x, y=0):
            self.encoder.fit(x)
            return self
        def transform(self, x, y=0):
            return self.encoder.transform(x)
        def get_params(self, y=0, deep=True):
            return self.encoder.get_params() 

    transformer = ColumnTransformer([
            #('quant', QuantileTransformer(), rob_cols),
            #('maxabs', MaxAbsScaler(), rob_cols),
            #('minmax', MinMaxScaler(), num_cols),
            #('rob', RobustScaler(), rob_cols),
            ('stand', StandardScaler(), num_cols),
            ('artists_enc', MyMultiLabelBinarizer(sparse_output=True), art_cols)
        ], 
        remainder='drop'
    )

transformer.fit_transform(X_train.head()).shape, \
transformer.transformers[0][1].fit_transform(X_train[num_cols].head()).shape, \
transformer.transformers[1][1].fit_transform(X_train.artists.head()).shape


# -------------------- Model selection --------------------
f1_macro = make_scorer(f1_score, average="macro")

if args.model == "lighgbm":
    # 1) LightGBM classifier
    clf = Pipeline([
        ('trans', transformer),
        ('clf', lightgbm.LGBMClassifier())
    ])

    param_grid = {
        "clf__boosting_type": ["gbdt", "rf", "dart"],
        "clf__num_leaves": [16, 32, 64, 128, 256, 512],
        "clf__min_child_samples": [2, 5, 10, 20, 30], 
        "clf__class_weight": [None, "is_unbalance", "balanced"],
        "clf__n_jobs": [-1],
        "clf__n_estimators": [500, 1000],
        "clf__max_bin": [500, 1000],
        "clf__objective": ["binary"]     
    }

    lgbm_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring=f1_macro).fit(X_train, y_train)
    f1_score(y_valid, lgbm_search.predict(X_valid), average="macro")

    lgbm_search.best_params_

elif args.model == "mnb":
    # 2) Multinomial Naive Bayes
    clf = Pipeline([
        ('trans', transformer),
        ('clf', MultinomialNB())
    ])

    param_grid = {
        "clf__alpha": [1e-10, 1e-05, 1]
    }

    mnb_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring=f1_macro).fit(X_train, y_train)
    f1_score(y_valid, mnb_search.predict(X_valid), average="macro")

    mnb_search.best_params_

elif args.model == "ridge":
    # 3) Ridge classifier
    clf = Pipeline([
        ('trans', transformer),
        ('clf', RidgeClassifier())
    ])

    param_grid = {
        "clf__alpha": [0.0001, 0.001, 0.1]
    }

    ridge_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring=f1_macro).fit(X_train, y_train)
    f1_score(y_valid, ridge_search.predict(X_valid), average="macro")

elif args.model == "lsvc":
    # 4) Linear SVC 
    clf = Pipeline([
        ('trans', transformer),
        ('clf', LinearSVC())
    ])

    param_grid = {
        "clf__penalty":["l1", "l2"],
        "clf__dual": [False],
        "clf__loss": ["hinge", "squared_hinge"],
        "clf__tol": np.logspace(-3,2,6),
        "clf__C": [1,10,20],
        "clf__class_weight": [None, "balanced"],
        "clf__max_iter": [1000, 2000]
    }

    lsvc_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring=f1_macro).fit(X_train, y_train)
    f1_score(y_valid, lsvc_search.predict(X_valid), average="macro")

    lsvc_search.best_params_

elif args.model == "rfc":
    # 5) Random forest classifier
    clf = Pipeline([
        ('trans', transformer),
        ('clf', RandomForestClassifier())
    ])

    param_grid = {
        "clf__penalty":["l1", "l2"],
        "clf__dual": [False],
        "clf__loss": ["hinge", "squared_hinge"],
        "clf__tol": np.logspace(-3,2,6),
        "clf__C": [1,10,20],
        "clf__class_weight": [None, "balanced"],
        "clf__max_iter": [1000, 2000]
    }

    lsvc_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring=f1_macro).fit(X_train, y_train)
    f1_score(y_valid, lsvc_search.predict(X_valid), average="macro")


# -------------------- optional finetuning for robust scaler --------------------
if args.tuning_robust_scaler == True:
    lightgbm.LGBMClassifier(clf__boosting_type="gbdt", clf__num_leaves=128,
                                clf__min_child_samples=5,clf__class_weight="balanced",
                                clf__n_jobs=-1, clf__n_estimators=1000)

    # get the models to evaluate
    models = get_models(num_cols, art_cols)
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, df.drop(columns=["mode"]), df["mode"], f1_macro)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()


# --------------------  Graphs for the report --------------------
if args.graphs == True:
    # a) Mode distribution
    arr = np.array(df["key"])
    arr_stand = (arr - arr.mean())/arr.std()
    arr_rob = (arr - np.percentile(arr, 50))/(np.percentile(arr, 75) - np.percentile(arr, 25))
    arr_minmax = (arr - min(arr))/(max(arr) - min(arr))

    arr = pd.Series(arr)
    arr_stand = pd.Series(arr_stand)
    arr_rob = pd.Series(arr_rob)
    arr_minmax = pd.Series(arr_minmax)

    fig, ax = plt.subplots()

    arr.plot.kde(label="original")
    arr_stand.plot.kde(label="standard")
    arr_rob.plot.kde(label="robust")
    arr_minmax.plot.kde(label="min_max")
    ax.set_xlabel("Key")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("./fig/standardization.pdf")

    # b) Features importance
    ax = lightgbm.plot_importance(lgbm_search.best_estimator_['clf'],  max_num_features=10, 
                            xlabel="Gain", title="",
                            importance_type="gain", grid=False)

    ax.set_yticklabels(["speechiness", "duration_ms", "acousticness", "valence", "danceability",
                        "tempo","liveness","key","energy","loudness"])
    ax.set_xticklabels(["", 20000, "", 60000, "", 100000, "", 140000])


    plt.ylabel("Features", fontsize=10)
    plt.xlabel("Total gain", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("./fig/feat_imp.pdf")


# -------------------- Upload on platform -------------------- 
clf_eval = lgbm_search.fit(df.drop(['mode'], axis=1), df["mode"])
df_eval = pd.read_csv("second_dataset/eval.tsv", sep="\t")

# remove id column
df_eval.drop(columns=["id"], inplace=True)

# split artists into list of artists
df_eval["artists"] = divide_artists(df_eval["artists"])

# clean artists, and leave them as a string
df_eval["artists"] = clean_artists(df_eval["artists"])

# predict the mode
df_eval['Predicted'] = lgbm_search.predict(df_eval)

# save the result
df_eval['Predicted'].to_csv('submission.csv',index_label='Id')