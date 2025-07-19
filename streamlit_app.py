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
from sklearn.metrics import r2_score, accuracy_score, classification_report

st.set_page_config(layout="wide")

def rtl(text):
    return f'<div dir="rtl" style="text-align:right">{text}</div>'

st.markdown(rtl("<h1>📘 פרויקט: מה משפיע על ציוני תלמידים?</h1>"), unsafe_allow_html=True)

st.markdown(rtl("""
הרבה תלמידים מאשימים גורמים כמו החוסר במורה פרטית, מקום מגורים והשכלת ההורים בציוניהם הגרועים.  
בפרויקט זה נראה איזה נתונים משפיעים ואילו לא משפיעים על הציון הסופי.

תלמידים רבים מאשימים את אי הצלחתם בלימודים בגורמים שמונים כמו החוסר במורה פרטי ועזרה מהבית.  
בפרויקט זה נראה מה באמת גורם לתלמיד לקבל ציונים יותר טובים.
"""), unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("SAP-4000.csv")
    df.dropna(inplace=True)
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Tutoring'] = df['Tutoring'].map({'No': 0, 'Yes': 1})
    df['Region'] = df['Region'].map({'Rural': 0, 'Urban': 1})
    df['Parent Education'] = df['Parent Education'].map({'Tertiary': 2, 'Primary': 0, 'Secondary': 1})
    df['Grade'] = df['Exam_Score'].apply(lambda x: 'Pass' if x >= 55 else 'Fail')
    return df

df = load_data()

st.markdown(rtl("<h2>🔎 ניתוח גרפי</h2>"), unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(rtl("ציון לפי מין"), unsafe_allow_html=True)
    sns.barplot(x='Gender', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

with col2:
    st.markdown(rtl("ציון לפי מגורים (עיר/כפר)"), unsafe_allow_html=True)
    sns.boxplot(x='Region', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

col3, col4 = st.columns(2)
with col3:
    st.markdown(rtl("ציון לפי שעות תרגול שבועי"), unsafe_allow_html=True)
    sns.scatterplot(x='HoursStudied/Week', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

with col4:
    st.markdown(rtl("ציון לפי השכלת הורים"), unsafe_allow_html=True)
    sns.barplot(x='Parent Education', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

def atc(att):
    if att < 60: return 'נמוכה מ-60%'
    elif att < 80: return 'בינונית 60-80%'
    return 'גבוהה מ-80%'

df['Attendance Group'] = df['Attendance(%)'].apply(atc)

st.markdown(rtl("ציון לפי נוכחות בשיעורים"), unsafe_allow_html=True)
sns.boxplot(x='Attendance Group', y='Exam_Score', data=df)
st.pyplot(plt.gcf()); plt.clf()

st.markdown(rtl("מתאם בין משתנים לציון במבחן"), unsafe_allow_html=True)
sns.heatmap(df.corr(numeric_only=True)[['Exam_Score']].T, annot=True, cmap='Blues')
st.pyplot(plt.gcf()); plt.clf()

st.markdown(rtl("<h2>🤖 המודלים שבניתי</h2>"), unsafe_allow_html=True)

X = df.drop(['Exam_Score', 'Grade', 'Attendance Group'], axis=1)
y_reg = df['Exam_Score']
y_cls = df['Grade']

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_cls_scaled = scaler.fit_transform(X_train_cls)
X_test_cls_scaled = scaler.transform(X_test_cls)

reg = LinearRegression().fit(X_train_scaled, y_train)
r2_lr = r2_score(y_test, reg.predict(X_test_scaled))

knn_pipe = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsRegressor())])
knn_cv = GridSearchCV(knn_pipe, {'knn__n_neighbors': list(range(1, 15))}, cv=5)
knn_cv.fit(X_train_scaled, y_train)
best_knn = knn_cv.best_estimator_
r2_knn = r2_score(y_test, best_knn.predict(X_test_scaled))

knn_clf = KNeighborsClassifier(n_neighbors=5).fit(X_train_cls_scaled, y_train_cls)
svm_model = SVC(kernel="linear").fit(X_train_cls_scaled, y_train_cls)
acc_knn = knn_clf.score(X_test_cls_scaled, y_test_cls)
acc_svm = svm_model.score(X_test_cls_scaled, y_test_cls)

st.markdown(rtl(f"Linear Regression: R² = {r2_lr:.2f}"), unsafe_allow_html=True)
st.markdown(rtl(f"KNN Regressor: R² = {r2_knn:.2f}, k = {knn_cv.best_params_['knn__n_neighbors']}"), unsafe_allow_html=True)
st.markdown(rtl(f"KNN Classifier: דיוק = {acc_knn:.2%}"), unsafe_allow_html=True)
st.markdown(rtl(f"SVM Classifier: דיוק = {acc_svm:.2%}"), unsafe_allow_html=True)

st.markdown(rtl("<h2>🔮 ניבוי לפי קלט</h2>"), unsafe_allow_html=True)

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
        X_input = scaler.transform(input_data[X.columns])

        st.markdown(rtl(f"Linear Regression חוזה ציון: {reg.predict(X_input)[0]:.2f}"), unsafe_allow_html=True)
        st.markdown(rtl(f"KNN Regressor חוזה ציון: {best_knn.predict(X_input)[0]:.2f}"), unsafe_allow_html=True)
        st.markdown(rtl(f"KNN Classifier חוזה הצלחה/כישלון: {knn_clf.predict(X_input)[0]}"), unsafe_allow_html=True)
        st.markdown(rtl(f"SVM Classifier חוזה הצלחה/כישלון: {svm_model.predict(X_input)[0]}"), unsafe_allow_html=True)

st.markdown("---")
st.markdown(rtl("📝 מסקנה: גורמים כמו נוכחות, תרגול והשקעה משפיעים הרבה יותר על הצלחת תלמיד מאשר מין, אזור מגורים או מורה פרטי."), unsafe_allow_html=True)
