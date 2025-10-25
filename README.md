# 🚗 Car Price Prediction Model - Performance Analysis

This project provides an **interactive analysis** of the results of a car price prediction model.  
Built with Streamlit, it allows you to visualize **prediction accuracy**, **error percentages**, and **overall model performance**.

---

## 🧠 Project Objective

The main goal of this project is to present the results of a car price prediction model in a **visual and analytical format**.  
Through this dashboard, users can:
- Monitor the model’s prediction accuracy  
- Compare real and predicted prices  
- Evaluate model performance based on error rates  
- Download the filtered results as a CSV file

---

## ⚙️ Technologies Used

| Technology | Description |
|-------------|-------------|
| **Python** | Core programming language |
| **Streamlit** | Interactive web application framework |
| **Pandas / NumPy** | Data processing and analysis |
| **Plotly** | Interactive visualization and graphing |
| **Scikit-learn** | Performance metrics (MAE, R², etc.) |

---

## 📂 Project Structure

```
📦 Car-Price-Prediction
├── app.py               # Streamlit dashboard (main interface)
├── analysis.ipynb       # Jupyter notebook for data/model analysis
├── requirements.txt     # Project dependencies
└── README.md            # Documentation (this file)
```

---

## 🚀 Installation & Run Guide

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<razuldev>/car-price-prediction.git
cd car-price-prediction
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Application Sections

### 🔹 Overview
- Scatter plot comparing real vs. predicted prices  
- Perfect prediction reference line  
- Distribution of prediction quality levels

### 🔹 Results Table
- Interactive and filterable table of results  
- Filtering by quality and price range  
- CSV export option

### 🔹 Car Detail View
- Individual car’s real and predicted prices  
- Absolute and percentage error metrics  
- Visual comparison chart

---

## 📈 Key Metrics

| Metric | Description |
|---------|-------------|
| **MAE (Mean Absolute Error)** | Average absolute difference (in AZN) |
| **MAPE (%)** | Mean Absolute Percentage Error |
| **R² (R-squared)** | Model explanatory power |
| **Accuracy (±5%)** | Percentage of predictions within ±5% error range |

---
