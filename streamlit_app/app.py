# importing nltk, pandas, pymorphy2, re for data manipulation
# importing os, PIL for images and working with files' path
import os
import pickle
import re

import nltk
import pandas as pd
import pymorphy2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from nltk.corpus import stopwords

# -- downloading russian stopwords for preprocessing
nltk.download('stopwords')
stopwords = stopwords.words('russian')

# -- finding an absolute path of files for the correct local work
path = os.path.dirname(__file__)

# -- opening necessary files
score_table = pd.read_csv(f'{path}/score_table.csv')
img = Image.open(f'{path}/school_dataset_example.png')

# -- class for russian text normalization
morph = pymorphy2.MorphAnalyzer()


# -- text normalization
def normalized(text):
    no_punctuation = re.sub(r'[^А-Яа-я]+', ' ', text.lower()).split(' ')
    norm_sentence = [morph.parse(word)[0].normal_form
                     for word in no_punctuation
                     if word not in stopwords]
    return ' '.join(norm_sentence)


# -- set page config
st.set_page_config(page_title='School website or not', page_icon=":books:")

# -- set main description
st.title('School website or not?')

# -- creating sidebar
with st.sidebar:
    # -- set data-type
    select_data = st.selectbox('Тип обработки данных:',
                               ['preprocessed', 'non-prepocessed'])

    # -- set vectorization type
    select_vector = st.selectbox('Тип векторизации текста:',
                                 ['BOW', 'BOW 3q-grams', 'TF-IDF', 'TF-IDF 4q-grams'])

    # -- set model-type
    select_model = st.selectbox('Тип моделей классификации:',
                                ['LogReg', 'NB', 'KNN', 'SVM'])

    # -- technology stack
    st.write('**Стэк технологий**: Python, Pandas, Sklearn, NLTK, pymoprhy2, Streamlit')
    st.write('В соответствующей ветке репозитория GitHub выложены notebook-и со всеми этапами проекта '
             '(не вошедшими в итоговое приложение)')

# -- making tabs
InfoTab, FunctionTab, Plot = st.tabs(["Информация", "Классификация", "Кластеризация данных"])

# -- set main tab information
with InfoTab:
    st.subheader('Что тут происходит?')

    st.markdown('Приложение создано для демонстрации работы некоторых **классических алгоритмов машинного обучения**.'
                ' Оно сделано как идейное продолжение хакатона Летней Школы с направления [**ML&Text**]('
                'https://www.letnyayashkola.org/nlp/) '

                ' и служит визуальным представлением небольшого pet-проекта с соответствующего репозитория на GitHub.')
    st.markdown('В обучающем датасете присутствовали тексты с архивированных веб-сайтов _(3630 примеров)_. '
                ' Тексты сильно загрязнены символами экранирования, html разметкой и знаками препинания.'
                ' Для каждого текста есть своя метка, обозначающая принадлежность к школьному сайту: если да ('
                '**:green[1]**), если нет (**:red[0]**)')

    # -- set downloaded image
    st.image(img, caption='Пример первых пяти строк датасета')

    st.markdown('К сожалению, пока нет разрешения публиковать обучающий датасет, '
                'потому в открытый доступ выложены только заранее предобученные модели.')

    st.markdown('Можно выбрать:')

    st.markdown("""
                * **preprocessed** - обработанные данные из датасета
                * **non-preprocessed** - необработанные данные из датасета
                """)

    st.markdown(
        'Предобработка текста заключалась в удалении всей латиницы (как способ борьбы с оставшейся html-разметкой), '
        'удалении пунктуации, приведении к нижнему регистру и лемматизации.')

    st.markdown('Способы векторизации текста:')
    st.markdown("""
                * [**BOW**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer)
                * [**BOW**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer) [**3q-grams**](https://en.wikipedia.org/wiki/N-gram) - здесь используются q-граммы в диапазоне до 3
                * [**TF-IDF**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
                * [**TF-IDF**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) [**4q-grams**](https://en.wikipedia.org/wiki/N-gram) - здесь используются q-граммы в диапазоне до 4
    """)

    st.markdown('Модели классификации:')

    st.markdown("""
                * [**LogReg**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - логистическая регрессия с **max_iter**=2000
                * [**NB**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - многоклассовый наивный байес
                * [**KNN**](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) - метод k-ближайших соседей
                * [**SVM**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - метод опорных векторов с линейным ядром
    """)

    st.write('Для всех возможных комбинаций уже заранее подсчитаны метрики на обучающем датасете '
             '(средние значения на [кросс-валидации](https://scikit-learn.org/stable/modules/cross_validation.html) '
             'по 5 батчей):')

    # -- set downloaded score dataframe
    st.dataframe(score_table)


# -- set plot tab with clusterization
with Plot:
    st.write(f'Все данные из обучающего датасета, которые не подходили в категорию "шольный сайт", были '
             f'кластеризированы в 20 кластеров (график может долго подгружаться). Подробнее про процесс '
             f'кластеризации можно почитать в соответствующем notebook в репозитории проекта (небольшой '
             f'статистический анализ текста, нахождение оптимального числа кластеров, векторизация и уменьшение '
             f'размерностей данных, визуализация).')

    # -- html plot
    plot_path = path + f'/clusterization_plot_copy.html'
    with open(plot_path, 'r') as file:
        plot = file.read()
    st.components.v1.html(plot, height=800)

# -- set function tav
with FunctionTab:
    st.write("")
    st.markdown('Напишите какой-нибудь текст (желательно на русском), который, как вы думаете, может встретиться на '
                'сайте школы. Подобранная модель попытается его классифицровать и определить, может ли он быть на '
                'школьном сайте.')
    st.write("")

    # -- set form for user's text
    with st.form(key="my_form"):

        text_template = 'Общеобразовательная школа была открыта в 1982 году. Школа располагается в здании, ' \
                        'построенном в этом же году. Исторически сложилось, что долгое время здание школы было ' \
                        'передано для организации дополнительного образования школьников, и только в сентябре 2000 ' \
                        'года коллектив школы начал свою работу в старых стенах. В здании имеется 46 учебных ' \
                        'кабинетов, методический кабинет, спортзал, компьютерный класс, библиотека и читальный зал. '

        # -- set text window
        text = st.text_area(
            "Введите свой текст",
            text_template,
            height=170,
            help="Для большей точности старайтесть вводить как можно более распространенные предложения."
        )

        # -- set submit button
        submit_button = st.form_submit_button(label='Получить предсказание')

        # -- checking all cases of valid inputs
        if "valid_inputs_received" not in st.session_state:
            st.session_state["valid_inputs_received"] = False

        if not submit_button and not st.session_state.valid_inputs_received:
            st.stop()

        elif submit_button and not text:
            st.warning("Вы ничего не написали :(")
            st.session_state.valid_inputs_received = False
            st.stop()

        # -- if everything is fine
        elif submit_button or st.session_state.valid_inputs_received:
            if submit_button:
                st.session_state.valid_inputs_received = True

                # -- finding models' names
                vec_model_path = path + f'/models/{select_vector} {select_data}.pkl'
                model_path = path + f'/models/{select_model} {select_vector} {select_data}.pkl'

                # -- open models
                with open(vec_model_path, 'rb') as vec_model, open(model_path, 'rb') as model:
                    vec_model = pickle.load(vec_model)
                    model = pickle.load(model)

                # -- if user chooses to preprocess data
                if select_data == 'preprocessed':
                    text = normalized(text)
                    st.divider()

                    # -- checking of valid inputs after preprocessing
                    if not text.replace(' ', ''):
                        st.warning('К сожалению, предобработка текста привела к пустой строке :(')
                    else:
                        st.write(f'Часть предобработанного вами текста: **:blue[{" ".join(text.split()[:10])}]**')

                st.divider()

                # -- text vectorization and making prediction
                prediction = int(model.predict(vec_model.transform([text])))

                st.subheader('Prediction')

                # -- cases of prediction
                if prediction:
                    st.success(f'**{prediction}**: Классификатор считает, что ваш текст может оказаться на **школьном '
                               f'сайте**! ✅')
                else:
                    st.warning(f'**{prediction}**: Увы, вряд ли это что-то, что вы найдете на **школьном сайте** ❌')

                # -- showing metrics for user's model
                st.write(f'Выбранная вами модель **{select_model} {select_vector} {select_data}** на обучающем '
                         f'датасете имеет следующие **метрики** (средние значения на [кросс-валидации]('
                         f'https://scikit-learn.org/stable/modules/cross_validation.html) по 5 батчей):')

                # -- selecting necessary row from score dataframe
                needed_row = score_table[(score_table['embeddings'] == select_vector)
                                         & (score_table['classifier'] == select_model)
                                         & (score_table['normalization'] == select_data)]

                # -- visualizing dataframe's row without columns with model's name
                st.dataframe(needed_row[score_table.keys()[3:]])
