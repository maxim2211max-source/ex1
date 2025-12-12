import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline  # Для загрузки Pipeline

# Загрузка модели (теперь VotingClassifier в Pipeline)
@st.cache_resource
def load_model():
    with open('voting_model.pkl', 'rb') as f:  # Изменено на voting_model.pkl
        model = pickle.load(f)
    return model

model = load_model()

# Заголовок приложения
st.title("Предсказание подписки на депозит (Bank Marketing)")
st.write("Введите данные клиента для предсказания вероятности подписки на депозит.")

# Определение фичей (как в модели)
numeric_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Создание формы для ввода (без изменений)
st.header("Введите фичи клиента:")

# Числовые фичи
age = st.number_input("Возраст (age)", min_value=18, max_value=100, value=30)
balance = st.number_input("Баланс счета (balance)", min_value=-10000, max_value=100000, value=1000)
day = st.number_input("День месяца (day)", min_value=1, max_value=31, value=15)
campaign = st.number_input("Количество контактов в кампании (campaign)", min_value=1, max_value=50, value=1)
pdays = st.number_input("Дни с последнего контакта (pdays)", min_value=-1, max_value=999, value=-1)
previous = st.number_input("Количество предыдущих контактов (previous)", min_value=0, max_value=50, value=0)

# Категориальные фичи (выпадающие списки)
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
job = st.selectbox("Работа (job)", options=job_options)

marital_options = ['divorced', 'married', 'single', 'unknown']
marital = st.selectbox("Семейное положение (marital)", options=marital_options)

education_options = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown']
education = st.selectbox("Образование (education)", options=education_options)

default_options = ['no', 'yes']
default = st.selectbox("Есть ли просрочка кредита (default)", options=default_options)

housing_options = ['no', 'yes']
housing = st.selectbox("Есть ли жилищный кредит (housing)", options=housing_options)

loan_options = ['no', 'yes']
loan = st.selectbox("Есть ли личный кредит (loan)", options=loan_options)

contact_options = ['cellular', 'telephone', 'unknown']
contact = st.selectbox("Тип контакта (contact)", options=contact_options)

month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month = st.selectbox("Месяц последнего контакта (month)", options=month_options)

poutcome_options = ['failure', 'nonexistent', 'success', 'unknown']
poutcome = st.selectbox("Результат предыдущей кампании (poutcome)", options=poutcome_options)

# Кнопка предсказания
if st.button("Предсказать"):
    # Создание DataFrame из ввода
    input_data = pd.DataFrame({
        'age': [age],
        'balance': [balance],
        'day': [day],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'poutcome': [poutcome]
    })
    
    # Предсказание вероятности (класс 1: 'yes')
    try:
        proba = model.predict_proba(input_data)[:, 1][0]  # Вероятность подписки
        st.success(f"Вероятность подписки на депозит: {proba:.4f}")
        if proba > 0.5:
            st.write("Рекомендация: Клиент склонен подписаться!")
        else:
            st.write("Рекомендация: Клиент, вероятно, не подпишется.")
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}. Проверьте модель и данные.")

# Футер
st.write("---")
st.write("Приложение на базе модели VotingClassifier (XGBoost + LogisticRegression). Данные из Kaggle Bank Marketing Dataset.")
