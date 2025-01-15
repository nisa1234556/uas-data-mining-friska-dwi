import streamlit as st    
import pandas as pd    
import numpy as np    
import matplotlib.pyplot as plt    
import seaborn as sns    
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler, OneHotEncoder    
from sklearn.compose import ColumnTransformer    
from sklearn.pipeline import Pipeline    
from sklearn.linear_model import LinearRegression    
from sklearn.neural_network import MLPRegressor    
from sklearn.ensemble import VotingRegressor    
from sklearn.metrics import mean_squared_error, r2_score    
  
# Load dataset    
def load_data():    
    file_path = 'Regression.csv'  # Update with the appropriate path if needed    
    data = pd.read_csv(file_path)    
    return data    
  
# Preprocessing pipeline    
def preprocess_data(data):    
    X = data.drop(columns=['charges'])    
    y = data['charges']    
  
    # Define preprocessing steps    
    numeric_features = ['age', 'bmi', 'children']    
    numeric_transformer = StandardScaler()    
  
    categorical_features = ['sex', 'smoker', 'region']    
    categorical_transformer = OneHotEncoder(drop='first')    
  
    preprocessor = ColumnTransformer(    
        transformers=[    
            ('num', numeric_transformer, numeric_features),    
            ('cat', categorical_transformer, categorical_features)    
        ]    
    )    
  
    return X, y, preprocessor    
  
# Main function    
def main():    
    st.set_page_config(page_title="Regresi Biaya Asuransi", page_icon="💰", layout="wide")    
    st.title("💻 Regresi Biaya Asuransi")    
    st.markdown("""    
    <style>    
    .stApp {    
        background-image: linear-gradient(to right, #ff7e5f, #feb47b);    
        color: white;    
    }    
    .sidebar .sidebar-content {    
        background: #2E3B4E;    
    }    
    </style>    
    """, unsafe_allow_html=True)    
  
    # Load and display dataset    
    data = load_data()    
    st.write("### 📊 Dataset", data.head())    
  
    # Preprocess data    
    X, y, preprocessor = preprocess_data(data)    
  
    # Split dataset    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
  
    # Sidebar layout    
    st.sidebar.title("⚙️ Model Configuration")    
    model_type = st.sidebar.selectbox("Pilih Algoritma", ["Linear Regression", "Neural Network", "Combined Model"])    
  
    if model_type == "Linear Regression":    
        model = Pipeline([    
            ('preprocessor', preprocessor),    
            ('regressor', LinearRegression())    
        ])    
    elif model_type == "Neural Network":    
        model = Pipeline([    
            ('preprocessor', preprocessor),    
            ('regressor', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))    
        ])    
    elif model_type == "Combined Model":    
        model = Pipeline([    
            ('preprocessor', preprocessor),    
            ('regressor', VotingRegressor([    
                ('lr', LinearRegression()),    
                ('nn', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))    
            ]))    
        ])    
  
    # Train model    
    model.fit(X_train, y_train)    
  
    # Predict and evaluate    
    y_pred = model.predict(X_test)    
    mse = mean_squared_error(y_test, y_pred)    
    r2 = r2_score(y_test, y_pred)    
  
    st.write(f"### 📈 Hasil Evaluasi ({model_type})")    
    st.write(f"- **Mean Squared Error:** {mse:.2f}")    
    st.write(f"- **R² Score:** {r2:.2f}")    
  
    # Input prediction    
    st.write("### 🔮 Prediksi Baru")    
    col1, col2 = st.columns(2)    
  
    with col1:    
        age = st.number_input("Usia", min_value=0, max_value=100, value=30)    
        sex = st.selectbox("Jenis Kelamin", ["male", "female"])    
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)    
  
    with col2:    
        children = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0)    
        smoker = st.selectbox("Status Merokok", ["yes", "no"])    
        region = st.selectbox("Wilayah", data['region'].unique())    
  
    input_data = pd.DataFrame({    
        'age': [age],    
        'sex': [sex],    
        'bmi': [bmi],    
        'children': [children],    
        'smoker': [smoker],    
        'region': [region]    
    })    
  
    if st.button("Prediksi Biaya Asuransi"):    
        prediction = model.predict(input_data)[0]    
        st.success(f"### 💵 Prediksi Biaya Asuransi: ${prediction:.2f}")    
  
    # Visualisasi Data    
    st.write("### 📊 Visualisasi Data")    
    fig, ax = plt.subplots()    
    sns.histplot(data['charges'], bins=30, kde=True, ax=ax)    
    ax.set_title('Distribusi Biaya Asuransi')    
    ax.set_xlabel('Biaya Asuransi')    
    ax.set_ylabel('Frekuensi')    
    st.pyplot(fig)    
  
    # Visualisasi berdasarkan model  
    st.write("### 📊 Visualisasi Hasil Model")  
    fig, ax = plt.subplots()  
    sns.regplot(x=y_test, y=y_pred, ax=ax)  
    ax.set_title(f'Prediksi vs Realita ({model_type})')  
    ax.set_xlabel('Biaya Asuransi Aktual')  
    ax.set_ylabel('Biaya Asuransi Prediksi')  
    st.pyplot(fig)  
  
    # Download dataset    
    st.write("### 📥 Unduh Dataset")    
    csv = data.to_csv(index=False).encode('utf-8')    
    st.download_button("Unduh Dataset", csv, "dataset.csv", "text/csv")    
  
    # Footer    
    st.markdown("---")    
    st.write("Aplikasi ini dibuat untuk membantu memprediksi biaya asuransi berdasarkan data yang diberikan. Terima kasih telah menggunakan aplikasi ini!")    
  
if __name__ == "__main__":    
    main()    
