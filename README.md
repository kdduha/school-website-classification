# school-website-classification

<div id="badges">
    <a href="https://www.python.org">
        <img src="https://img.shields.io/badge/python-6a6a6a?style=flat&logo=python&logoColor=white" alt="python badge"/>
    </a>
    <a href="https://scikit-learn.org">
        <img src="https://img.shields.io/badge/sklearn-597b9a?style=flat&logo=sklearn&logoColor=white" alt="sklearn badge"/>
    </a>
    <a href="https://streamlit.io">
        <img src="https://img.shields.io/badge/sreamlit-e60d1a?style=flat&logo=streamlit&logoColor=white" alt="streamlit badge"/>
    </a>
    <a href="https://pandas.pydata.org">
        <img src="https://img.shields.io/badge/pandas-7140ff?style=flat&logo=pandas&logoColor=white" alt="pandas badge"/>
    </a>
    <a href="https://www.nltk.org">
        <img src="https://img.shields.io/badge/nltk-%23042e3c?style=flat&logo=nltk&logoColor=white" alt="nltk badge"/>
    </a>
    <a href="https://pymorphy2.readthedocs.io">
        <img src="https://img.shields.io/badge/pymorphy2-5287ac?style=flat&logo=pymorphy2&logoColor=white" alt="pymorphy2 badge"/>
    </a>
    <a href="https://plotly.com">
        <img src="https://img.shields.io/badge/plotly-%231a1a1a?style=flat&logo=plotly&logoColor=white" alt="plotly badge"/>
    </a>
    <a href="https://matplotlib.org">
        <img src="https://img.shields.io/badge/matplotlib-3d85c6?style=flat&logo=matplotlib&logoColor=white" alt="matplotlib badge"/>
    </a>
</div>

**Pet-project**: comparison of different algorithms for school website classification + text clusterization + streamlit app

The final visualization in the form of a **streamlit web service** is [**here**](https://school-website-classification.streamlit.app/)
___

The pet-project was made on the basis of a small closed hackathon *(by this reason I've only published models' weights without training data)* of the Summer School from the **ML&Text track (2023 year)**.

The training dataset contained texts from archived websites *(3630 examples)*. The texts are heavily polluted with escape symbols, html markup and punctuation marks. Each text has its own label indicating belonging to the school website: if yes (1), if not (0).

**What did I do?**
- on the cross validation of 5 batches, I compared:
    - several classical classification algorithms: 
        - *Logistic Regression*
        - *Naive Baise*
        - *KNN (k-nearest neighbours)*
        - *Linear SVM (support vector machine)*
    - different vectorization models:
        - *BOW (Bag of Words), q-grams (and not)*
        - *TF-IDF, q-grams (and not)*
    - preprocessed and non-preprocessed data
<p> </p>

- I checked whether the quality of the model improves depending on the preprocessing of texts

- A large number of words in the data are separated into parts by spaces, so I checked whether *q-grams* can improve the metrics

- I did a frequency analysis of the data and compared it with *TF-IDF features*

- I tried to do clusterization of non-school websites:
    - I found an optimal number of clusters by different methods:
        - *AgglomerativeClustering with automatic determination of the optimal number of clusters*
        - *dendrogram of clusters*
        - *comparison of silhouette Score for different numbers of clusters*
    - I lowered the dimension of vector spaces by using *IncrementalPCA* and made an interactive scattering plot
<p> </p>

- I visualized all the work done in the [streamlit web service (ru)](https://school-website-classification.streamlit.app/) where using different pre-trained models you can determine whether the text is found on the school website and look at the metrics

In a folder **streamlit_app** are located requirements, a web-service code, training script and 32 classification models with different configurations.

In a folder **research** are stored my notebooks where I tested hypotheses, did a frequency analysis + words' clusterization and explained my decisions, problems, possible solutions and some troubles that I couldn't fix.
