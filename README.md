# school-website-classification

![python](https://img.shields.io/badge/python-6a6a6a?style=flat&logo=python&logoColor=white&link=https%3A%2F%2Fwww.python.org%2F)
![sklearn](https://img.shields.io/badge/sklearn-597b9a?style=flat&logo=sklearn&logoColor=white&link=https%3A%2F%2Fscikit-learn.org%2Fstable%2F)
![streamlit](https://img.shields.io/badge/sreamlit-e60d1a?style=flat&logo=streamlit&logoColor=white&link=https%3A%2F%2Fstreamlit.io%2F)
![pandas](https://img.shields.io/badge/pandas-7140ff?style=flat&logo=pandas&logoColor=white&link=https%3A%2F%2Fpandas.pydata.org%2Fdocs%2F%23)
![nltk](https://img.shields.io/badge/nltk-%23042e3c?style=flat&logo=nltk&logoColor=white&link=https%3A%2F%2Fwww.nltk.org%2F)
![pymorphy2](https://img.shields.io/badge/pymorphy2-5287ac?style=flat&logo=pymorphy2&logoColor=white&link=https%3A%2F%2Fpymorphy2.readthedocs.io%2Fen%2Fstable%2F)
![plotly](https://img.shields.io/badge/plotly-%231a1a1a?style=flat&logo=plotly&logoColor=white&link=https%3A%2F%2Fplotly.com%2Fpython%2F)
![matplotlib](https://img.shields.io/badge/matplotlib-3d85c6?style=flat&logo=matplotlib&logoColor=white&link=https%3A%2F%2Fmatplotlib.org%2F)

**Pet-project**: comparison of different algorithms for school website classification + text clusterization + streamlit app

The final visualization in the form of a **streamlit web service** is [**here**](https://school-website-classification.streamlit.app/)
___

The pet-project was made on the basis of a small closed hackathon *(by this reason I've only published models' weights without training data)* of the Summer School from the **ML&Text direction (2023 year)**.

The training dataset contained texts from archived websites *(3630 examples)*. The texts are heavily polluted with escape symbols, html markup and punctuation marks. Each text has its own label indicating belonging to the school website: if yes (1), if not (0).

**What did I do?**
- on the cross validation of 5 batches, I compared:
    - several classical classification algorithms: 
        - *Logistig Regression*
        - *Naive Baise*
        - *KNN (k-nearest neighbours)*
        - *Linear SVM (support vector machine)*
    - different vectorization models:
        - *BOW (Bag of Words), q-grams (and not)*
        - *TF-IDF, q-grams (and not)*
    - preprocessed and non-preprocessed data
<p> </p>

- I checked whether the quality of the model improves depending on the preprocessing of texts

- A large number of words in the data are separated into parts by spaces, so I checked whether q-grams can improve the metrics

- I did a frequency analysis of the data and compared it with TF-IDF features

- I tried to do clusterization of non-school websites:
    - I found an optimal number of clusters by different methods:
        - *AgglomerativeClustering with automatic determination of the optimal number of clusters*
        - *dendrogram of clusters*
        - *comparison of silhouette Score for different numbers of clusters*
    - I lowered the dimension of vector spaces by using IncrementalPCA and made an interactive scattering plot
<p> </p>

- I visualized all the work done in the [streamlit web service](https://school-website-classification.streamlit.app/) where using different pre-trained models you can determine whether the text is found on the school website and look at the metrics

In a folder **streamlit_app** are located requirenments, a web-service code, training script and 32 classification models with different configuration.

In a folder **research** are stored my notebooks where I tested hypotheses, did a frequency analysis + words' clusterization and explained my decisions, problems, possible solutions and some troubles that I couldn't fix.
