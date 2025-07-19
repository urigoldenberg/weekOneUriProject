import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import plotly.express as px

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

st.title("📚 פרויקט ציוני תלמידים - ניתוח ותחזית")

menu = st.sidebar.radio("תפריט", ["הצגת הדאטה", "גרפים", "מודלים", "השוואת מודלים"])

if menu == "הצגת הדאטה":
    st.subheader("הדאטה המלא")
    st.dataframe(df)

elif menu == "גרפים":
    st.subheader("גרפים חשובים")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="Gender", y="Exam_Score", title="ציון לפי מין")
        st.plotly_chart(fig)
    with col2:
        fig = px.box(df, x="Region", y="Exam_Score", title="ציון לפי אזור")
        st.plotly_chart(fig)
    fig = px.scatter(df, x="HoursStudied/Week", y="Exam_Score", title="זמן לימוד מול ציון")
    st.plotly_chart(fig)

elif menu == "מודלים":
    st.subheader("מודלים לחיזוי הצלחה")
    model_type = st.selectbox("בחר מודל", ["Linear Regression", "KNN", "SVM"])

    X = df.drop(["Exam_Score", "Grade"], axis=1)
    y_class = df['Grade']
    y_reg = df['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y_class if model_type != "Linear Regression" else y_reg,
                                                        test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "Linear Regression":
        reg = LinearRegression()
        reg.fit(X_train_scaled, y_train)
        score = reg.score(X_test_scaled, y_test)
        st.metric("R²", f"{score:.2f}")

    elif model_type == "KNN":
        pipe = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
        pipe.fit(X_train_scaled, y_train)
        y_pred = pipe.predict(X_test_scaled)
        st.text("דוח סיווג:")
        st.code(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_estimator(pipe, X_test_scaled, y_test, cmap="Blues")
        st.pyplot(plt.gcf())

    elif model_type == "SVM":
        svm_model = SVC(kernel="linear")
        svm_model.fit(X_train_scaled, y_train)
        y_pred = svm_model.predict(X_test_scaled)
        st.text("דוח סיווג:")
        st.code(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_estimator(svm_model, X_test_scaled, y_test, cmap="Purples")
        st.pyplot(plt.gcf())

elif menu == "השוואת מודלים":
    st.subheader("השוואת KNN מול SVM")

    X = df.drop(["Exam_Score", "Grade"], axis=1)
    y = df["Grade"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    svm = SVC(kernel="linear")
    svm.fit(X_train_scaled, y_train)

    st.metric("🎯 דיוק KNN", f"{accuracy_score(y_test, knn.predict(X_test_scaled)):.2f}")
    st.metric("🎯 דיוק SVM", f"{accuracy_score(y_test, svm.predict(X_test_scaled)):.2f}")
