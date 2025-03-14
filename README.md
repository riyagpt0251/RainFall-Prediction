---

# 🌦️ Weather Data Prediction with Machine Learning  
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)  
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikitlearn)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?style=flat-square&logo=plotly)  

🔗 **Live Demo:** [Google Colab Notebook](https://colab.research.google.com/)  

---

## 📌 Overview  

This project utilizes **Machine Learning (Linear Regression)** to analyze and predict **temperature trends** based on weather data. The dataset includes attributes like **humidity, wind speed, pressure, and visibility**, which are used to train the model.  

- 📊 **Dataset:** `1. Weather Data.csv`
- 📉 **Model:** `Linear Regression`
- 📌 **Prediction Target:** `Temperature (Temp_C)`
- 🔬 **Tools Used:** `Pandas, NumPy, Scikit-Learn, Matplotlib`
- 📁 **Output Model:** `rainfall_predictor_model.pkl`  

---

## 📂 Table of Contents  

- [📌 Overview](#-overview)  
- [📂 Dataset](#-dataset)  
- [📊 Data Preprocessing](#-data-preprocessing)  
- [🤖 Machine Learning Model](#-machine-learning-model)  
- [📈 Results & Evaluation](#-results--evaluation)  
- [💾 Save & Load Model](#-save--load-model)  
- [📸 Visualizations](#-visualizations)  
- [📌 Installation & Usage](#-installation--usage)  
- [🤝 Contributing](#-contributing)  
- [📜 License](#-license)  

---

## 📂 Dataset  

The dataset contains weather data with the following columns:  

| Date/Time       | Temp_C | Humidity (%) | Wind Speed (km/h) | Visibility (km) | Pressure (kPa) | Weather Conditions |
|----------------|--------|--------------|-------------------|----------------|--------------|------------------|
| 1/1/2012 0:00 | -1.8°C | 86           | 4                 | 8.0            | 101.24       | Fog              |
| 1/1/2012 1:00 | -1.8°C | 87           | 4                 | 8.0            | 101.24       | Fog              |
| 1/1/2012 2:00 | -1.8°C | 89           | 7                 | 4.0            | 101.26       | Freezing Drizzle,Fog |

🔹 **Target Variable:** `Temp_C (Temperature in Celsius)`  

---

## 📊 Data Preprocessing  

The dataset undergoes the following preprocessing steps:  

✔️ **Handling Missing Values:**  
   - Missing numeric values replaced with column means.  

✔️ **Feature Engineering:**  
   - `Date/Time` converted into `Year, Month, Day, Hour`.  
   - `Weather` column converted into **One-Hot Encoding**.  

✔️ **Data Scaling:**  
   - Applied `StandardScaler()` to normalize features.  

---

## 🤖 Machine Learning Model  

### ✅ **Train-Test Split**  
The dataset is split into:  
- **80% Training Set**  
- **20% Test Set**  

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### ✅ **Model Training**  
We use **Linear Regression** to fit the data:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 📈 Results & Evaluation  

After training, we evaluate the model using **Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.**  

| Metric  | Value |
|---------|-------|
| **MSE** | `1.0084e-06` |
| **RMSE** | `0.0010` |
| **R² Score** | `0.0` |

📌 **Insights:**  
- The R² score is very low, meaning the model does not explain much of the variance in temperature.  
- Additional features may be required to improve predictions.  

---

## 💾 Save & Load Model  

The trained model is saved for future use:  

```python
import joblib
joblib.dump(model, 'rainfall_predictor_model.pkl')
```

To load the model later:  

```python
model = joblib.load('rainfall_predictor_model.pkl')
```

---

## 📸 Visualizations  

📌 **1️⃣ Distribution of Temperature**  
![Temperature Histogram](https://via.placeholder.com/800x400?text=Temperature+Distribution+Graph)  

📌 **2️⃣ Feature Importance**  
![Feature Importance](https://via.placeholder.com/800x400?text=Feature+Importance+Graph)  

📌 **3️⃣ Actual vs Predicted Temperature**  
![Prediction Comparison](https://via.placeholder.com/800x400?text=Actual+vs+Predicted+Graph)  

---

## 📌 Installation & Usage  

### 🔹 **1. Clone the Repository**  
```bash
git clone https://github.com/your-username/weather-prediction-ml.git
cd weather-prediction-ml
```

### 🔹 **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 🔹 **3. Run the Script**  
```bash
python weather_prediction.py
```

---

## 🤝 Contributing  

💡 Want to improve this project? Follow these steps:  

1. **Fork the Repository**  
2. **Create a New Branch (`feature-new-feature`)**  
3. **Commit Your Changes (`git commit -m 'Add new feature'`)**  
4. **Push to the Branch (`git push origin feature-new-feature`)**  
5. **Create a Pull Request**  

---

## 📜 License  

📄 This project is licensed under the **MIT License** – feel free to use and modify!  

📧 **Have Questions?** Feel free to contact me at [your-email@example.com](mailto:your-email@example.com)  

---

🔥 **Enjoy Coding!** 🚀✨  

---

This `README.md` file makes your GitHub project more **engaging, professional, and visually appealing!** 🚀
