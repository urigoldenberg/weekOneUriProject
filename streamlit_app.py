
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score, classification_report, ConfusionMatrixDisplay

st.set_page_config(layout="wide")

st.title("📊 פרויקט: מה משפיע על ציוני תלמידים?")

st.markdown("""
## 🧠 רקע
הרבה תלמידים מאשימים גורמים כמו החוסר במורה פרטית, מקום מגורים והשכלת ההורים בציוניהם הגרועים.  
בפרויקט זה נראה איזה נתונים משפיעים ואילו לא משפיעים על הציון הסופי.
""")

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

st.header("🔎 ניתוח גרפי")

col1, col2 = st.columns(2)
with col1:
    st.subheader("מין מול ציון")
    sns.barplot(x='Gender', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

with col2:
    st.subheader("מגורים עירוניים מול ציונים")
    sns.boxplot(x='Region', y='Exam_Score', data=df)
    st.pyplot(plt.gcf()); plt.clf()

st.subheader("זמן תרגול שבועי מול ציון")
sns.scatterplot(x='HoursStudied/Week', y='Exam_Score', data=df)
st.pyplot(plt.gcf()); plt.clf()

st.subheader("השכלת הורים מול ציון")
sns.barplot(x='Parent Education', y='Exam_Score', data=df)
st.pyplot(plt.gcf()); plt.clf()

st.subheader("נוכחות בשיעור מול ציון")
def atc(att):
    if att < 60:
        return 'Poor Attendance <60'
    elif att < 80:
        return 'Average Attendance 80<>60'
    else:
        return 'Excellent Attendance >80'

df['Attendance Group'] = df['Attendance(%)'].apply(atc)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attendance Group', y='Exam_Score', data=df)
st.pyplot(plt.gcf()); plt.clf()

st.subheader("מתאם מול ציון מבחן")
plt.figure(figsize=(17, 1))
sns.heatmap(df.corr(numeric_only=True)[['Exam_Score']].T, cmap="Blues", annot=True)
st.pyplot(plt.gcf()); plt.clf()

st.header("🤖 בניית מודלים")

X_reg = df.drop(['Exam_Score', 'Grade', 'Attendance Group'], axis=1)
y_reg = df['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.subheader("📈 Linear Regression")
reg = LinearRegression()
reg.fit(X_train_scaled, y_train)
y_pred_lr = reg.predict(X_test_scaled)
st.write(f"**R²:** {r2_score(y_test, y_pred_lr):.2f}")

st.subheader("📊 KNN Regressor")
knn_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('knn', KNeighborsRegressor())
])
param_grid = {'knn__n_neighbors': list(range(1, 15))}
knn_search = GridSearchCV(knn_pipe, param_grid=param_grid, cv=5)
knn_search.fit(X_train_scaled, y_train)
best_knn = knn_search.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)
st.write(f"**R²:** {r2_score(y_test, y_pred_knn):.2f}")

st.subheader("🎯 KNN Classifier")
X_cls = df.drop(['Exam_Score', 'Grade', 'Attendance Group'], axis=1)
y_cls = df['Grade']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
X_train_cls = scaler.fit_transform(X_train_cls)
X_test_cls = scaler.transform(X_test_cls)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_cls, y_train_cls)
y_pred_class = knn_classifier.predict(X_test_cls)
st.text("דו"ח סיווג:")
st.code(classification_report(y_test_cls, y_pred_class))
ConfusionMatrixDisplay.from_estimator(knn_classifier, X_test_cls, y_test_cls, cmap='Blues')
st.pyplot(plt.gcf()); plt.clf()

st.subheader("🧠 SVM Classifier")
svm_model = SVC(kernel="linear")
svm_model.fit(X_train_cls, y_train_cls)
y_pred_svm = svm_model.predict(X_test_cls)
st.text("דו"ח סיווג:")
st.code(classification_report(y_test_cls, y_pred_svm))
ConfusionMatrixDisplay.from_estimator(svm_model, X_test_cls, y_test_cls, cmap='Purples')
st.pyplot(plt.gcf()); plt.clf()

st.markdown("___")
st.markdown("*תלמידים רבים מאשימים את אי הצלחתם בגורמים חיצוניים. מטרת הפרויקט הייתה לבדוק לעומק האם באמת יש לכך השפעה, או שהשפעה משמעותית נובעת דווקא מגורמים אחרים כמו תרגול ונוכחות.*")
