# 🛒 Amazon Sales Analytics Dashboard
A fully interactive **Streamlit dashboard** that analyzes the Amazon Sales Dataset from Kaggle.  
Includes data cleaning, EDA, customer segmentation, collaborative filtering recommendations, and interactive visualizations.

---

## ⭐ Features

### 1️⃣ Dataset Overview
- Shows dataset structure and metadata  
- Includes cleaning + preprocessing with custom functions

### 2️⃣ Data Preview
- Displays the first 100 rows  
- Prices, ratings, and discounts are formatted for clarity

### 3️⃣ Summary Statistics
**Numerical statistics:**  
- Mean, median, skewness, missing values  

**Categorical statistics:**  
- Unique users  
- Unique products  
- Unique categories  
- Unique reviews  

### 4️⃣ Interactive Visualizations
Includes filters for:  
- Category  
- Price range  
- Discount %  
- Rating  

Visual insights generated:  
- Category distribution  
- Discount impact on ratings  
- Customer behavior patterns  
- Customer segmentation (PCA + cluster sizes)

### 5️⃣ Product Recommendation System
Uses **Collaborative Filtering** with:  
- User–item matrix  
- Cosine similarity  
- Top-N recommendations

### 6️⃣ Insight Section
Summarizes insights on:  
- Category performance  
- Discount behavior  
- Customer segments (Loyal Customers, Discount Chasers, etc.)

---

## 📦 Dependencies
- `streamlit` – UI  
- `pandas`, `numpy` – data manipulation  
- `matplotlib`, `seaborn` – visualizations  
- `scikit-learn` – PCA, KMeans clustering, scaling  
- `kagglehub` – dataset download
