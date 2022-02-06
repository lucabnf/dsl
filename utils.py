from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import lightgbm

# Function to remove undesired characters from artists
def cleaner(a):
    a = "".join([x for x in a if x not in "[]''" + '""']).lower()
    a = a.replace("/", ", ")
    a = a.replace('\\', ', ')
    a = a.replace(";", ", ")
    a = a.replace("-", ", ")
    a = a.replace("&", ", ")
    a = a.replace("+", ", ")
    a = a.replace('/[0-9]/g', '')
    a = a.replace('/\d+|^\s+|\s+$/g','');
    a = a.replace("', '", ",")
    a = a.replace(" ,", ", ")
    a = a.replace(",  ", ", ")
        
    return a

# Function to split "artists" to form a list of artists
def divide_artists(s):
    
    divided_s = []
    
    for el in s:
        temp = cleaner(el)  # (1)
          
        temp = temp.split(", ")   # (2)
        divided_s.append(temp)
        
    return list(divided_s)

# Function to clean a Series of artists
def clean_artists(s):
    
    clean_s = []
    
    for el in s:
        temp = cleaner(el)
        clean_s.append(temp)
    
    return clean_s

# optimized function to sum all the counter
def sum_counters(counter_list):

    if len(counter_list) > 10:

        counter_0 = sum_counters(counter_list[:int(len(counter_list)/2)])
        counter_1 = sum_counters(counter_list[int(len(counter_list)/2):])

        return sum([counter_0, counter_1], Counter())

    else:

        return sum(counter_list, Counter())

# --- Functions used for finetune the Robust Scaler ---
# get a list of models to evaluate
def get_models(num_cols, art_cols):
    models = dict()
    for value in [1, 5, 10, 15, 20, 25, 30]: #99th perc, 95th perc, 90th perc, ..., 70th perc
        # define the pipeline
        trans = ColumnTransformer([
            ('stand', RobustScaler(value, 100-value), num_cols),
            ('tfidf', TfidfVectorizer(stop_words='english', use_idf=False), art_cols)
        ], remainder='drop')
        model = lightgbm.LGBMClassifier(clf__boosting_type="gbdt", clf__num_leaves=128,
                            clf__min_child_samples=5,clf__class_weight="balanced",
                            clf__n_jobs=-1, clf__n_estimators=1000)
        models[str(value)] = Pipeline(steps=[('t', trans), ('m', model)])
    return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y, f1_macro):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring=f1_macro, cv=cv, n_jobs=-1, error_score='raise')
    return scores