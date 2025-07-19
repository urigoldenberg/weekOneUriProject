import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, classification_report, ConfusionMatrixDisplay
import numpy as np

st.set_page_config(layout="wide")

def rtl(text):
    return f'<div dir="rtl" style="text-align:right">{text}</div>'

st.markdown(rtl("## 📊 פרויקט: מה משפיע על ציוני תלמידים?"), unsafe_allow_html=True)

st.markdown(rtl("""
### 🧠 רקע  
הרבה תלמידים מאשימים גורמים כמו החוסר במורה פרטית, מקום מגורים והשכלת ההורים בציוניהם הגרועים.  
בפרויקט זה נראה איזה נתונים משפיעים ואילו לא משפיעים על הציון הסופי.
"""), unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("SAP-4000.csv")
    df.dropna(inplace=True)
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Tutoring'] = df['Tutoring'].map({'No': 0, 'Yes': 1})
    df['Region'] = df['Region'].map({'Rural': 0, 'Urban': 1})
    df['Parent Education'] = df['Parent Education'].map({'Tertiary': 2, 'Primary': 0,'Secondary':1})
    df['Grade'] = df['Exam_Score'].apply(lambda x: 'Pass' if x >= 55 else 'Fail')
    return df

df = load_data()

st.markdown(rtl("### 🔎 ניתוח גרפי"), unsafe_allow_html=True)

with st.expander("מין מול ציון"):
    plt.figure(figsize=(4, 3))
    sns.barplot(x='Gender', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

with st.expander("מגורים עירוניים מול ציונים"):
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='Region', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

with st.expander("⏱ זמן תרגול שבועי מול ציון"):
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='HoursStudied/Week', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

with st.expander("🎓 השכלת הורים מול ציון"):
    plt.figure(figsize=(4, 3))
    sns.barplot(x='Parent Education', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

with st.expander("📚 נוכחות בשיעור מול ציון"):
    def atc(att):
        if att < 60:
            return 'Poor Attendance <60'
        elif att < 80:
            return 'Average Attendance 60-80'
        else:
            return 'Excellent Attendance >80'
    df['Attendance Group'] = df['Attendance(%)'].apply(atc)
    plt.figure(figsize=(4, 3))
    sns.boxplot(x='Attendance Group', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

with st.expander("📈 מתאם מול ציון מבחן"):
    plt.figure(figsize=(6, 1))
    sns.heatmap(df.corr(numeric_only=True)[['Exam_Score']].T, cmap="Blues", annot=True)
    st.pyplot(plt.gcf()); plt.clf()

st.markdown(rtl("### 🤖 בניית מודלים"), unsafe_allow_html=True)

X_reg = df.drop(['Exam_Score', 'Grade', 'Attendance Group'], axis=1)
y_reg = df['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LinearRegression().fit(X_train_scaled, y_train)
y_pred_lr = reg.predict(X_test_scaled)
st.write(f"**R² לינארי:** {r2_score(y_test, y_pred_lr):.2f}")

knn_pipe = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsRegressor())])
knn_search = GridSearchCV(knn_pipe, {'knn__n_neighbors': list(range(1, 15))}, cv=5)
knn_search.fit(X_train_scaled, y_train)
best_knn = knn_search.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)
st.write(f"**R² KNN Regressor:** {r2_score(y_test, y_pred_knn):.2f}")

X_cls = df.drop(['Exam_Score', 'Grade', 'Attendance Group'], axis=1)
y_cls = df['Grade']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
X_train_cls = scaler.fit_transform(X_train_cls)
X_test_cls = scaler.transform(X_test_cls)

knn_classifier = KNeighborsClassifier(n_neighbors=5).fit(X_train_cls, y_train_cls)
svm_model = SVC(kernel="linear").fit(X_train_cls, y_train_cls)

st.markdown(rtl("### 🧪 ניבוי לפי קלט שלך"), unsafe_allow_html=True)

with st.form("prediction_form"):
    gender = st.selectbox("מין", ["Female", "Male"])
    tutoring = st.selectbox("מורה פרטי", ["No", "Yes"])
    region = st.selectbox("אזור מגורים", ["Rural", "Urban"])
    parent_edu = st.selectbox("השכלת הורים", ["Primary", "Secondary", "Tertiary"])
    hours = st.slider("שעות לימוד בשבוע", 0, 40, 10)
    attendance = st.slider("נוכחות בשיעור (%)", 0, 100, 85)
    submitted = st.form_submit_button("חשב תחזית")

    if submitted:
        input_data = pd.DataFrame([{
            'Gender': 1 if gender == "Male" else 0,
            'Tutoring': 1 if tutoring == "Yes" else 0,
            'Region': 1 if region == "Urban" else 0,
            'Parent Education': {"Primary": 0, "Secondary": 1, "Tertiary": 2}[parent_edu],
            'HoursStudied/Week': hours,
            'Attendance(%)': attendance
        }])
        X_input = scaler.transform(input_data)
        reg_pred = reg.predict(X_input)[0]
        knn_pred = best_knn.predict(X_input)[0]
        svm_pred = svm_model.predict(X_input)[0]

        st.markdown(rtl(f"📘 **Linear Regression חוזה ציון:** {reg_pred:.2f}"), unsafe_allow_html=True)
        st.markdown(rtl(f"📗 **KNN חוזה ציון:** {knn_pred:.2f}"), unsafe_allow_html=True)
        st.markdown(rtl(f"📕 **SVM חוזה הצלחה/כישלון:** {svm_pred}"), unsafe_allow_html=True)

st.markdown("---")
st.markdown(rtl("📝 **סיכום:** תלמידים רבים מאשימים את אי הצלחתם בגורמים חיצוניים. מטרת הפרויקט הייתה לבדוק לעומק האם באמת יש לכך השפעה, או שהשפעה משמעותית נובעת דווקא מגורמים אחרים כמו תרגול ונוכחות."), unsafe_allow_html=True)
