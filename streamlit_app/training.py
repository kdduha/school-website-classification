# importing pandas for reading data
import pandas as pd
# importing pickle for models' saving
import pickle

# importing sklearn for training
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

# -- reading datasets
df = pd.read_csv('../school.csv').dropna()
df_norm = pd.read_csv('../school_normalized.csv')

# -- dictionary for types of datasets
preprocessing = {'preprocessed': df_norm, 'non-prepocessed': df}

# -- dictionary for types of vectorization
vector_models = {'TF-IDF': TfidfVectorizer(),
                 'TF-IDF 4q-grams': TfidfVectorizer(ngram_range=(1, 4), analyzer='char'),
                 'BOW 3q-grams': CountVectorizer(ngram_range=(1, 3), analyzer='char'),
                 'BOW': CountVectorizer()}

seed = 420
# -- dictionary for types of models
models = {'LogReg': LogisticRegression(max_iter=2000, random_state=seed),
          'NB': MultinomialNB(),
          'KNN': KNeighborsClassifier(),
          'SVM': SVC(kernel='linear', random_state=seed)}

# -- choosing data type
for data_type in preprocessing:

    X = preprocessing[data_type]['main_page']
    Y = preprocessing[data_type]['school']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed,
                                                        shuffle=True, stratify=Y)

    # -- choosing vector type
    for vector_type in vector_models:

        vec = vector_models[vector_type]
        X_train_vec = vec.fit_transform(X_train)
        # X_test_vec = vec.transform(X_test)

        # saving vec model for future text transforming
        with open(f'./models/{vector_type} {data_type}.pkl', 'wb') as f:
            pickle.dump(vec, f)
            print(f'vec model {vector_type} {data_type}.pkl saved')

        # choosing model type
        for model_type in models:
            model = models[model_type]
            model.fit(X_train_vec, y_train)

            # saving classifier
            with open(f'./models/{model_type} {vector_type} {data_type}.pkl', 'wb') as f:
                pickle.dump(model, f)
                print(f'classifier {model_type} {vector_type} {data_type}.pkl saved')
