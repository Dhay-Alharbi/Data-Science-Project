Amazon Sales Analytics Dashboard

A fully interactive Streamlit dashboard that analyzes the Amazon Sales Dataset from Kaggle.
The dashboard provides data cleaning, EDA, customer segmentation, collaborative filtering recommendations, and interactive visualizations

 Features
1 Dataset Overview

Displays dataset features, structure, and source.

Cleaned and processed using custom functions.

2. Data Preview

View first 100 rows with formatted prices, ratings, and discount values.

3. Summary Statistics

Numerical summary (mean, median, skewness, missing values).

Categorical summary (unique users, products, reviews).

4. Interactive Visualizations

Includes filters for:
✔ Category
✔ Price range
✔ Discount %
✔ Rating

Provides visual insights such as:

Category distribution

Impact of discounts on ratings

Customer behavior patterns

Customer segmentation (PCA visualization + cluster size distribution)

5. Product Recommendation System

Uses Collaborative Filtering to recommend products based on user similarity.
Includes:

User–item matrix

Cosine similarity

Top‐N recommendations

6. Insight Section

Contains summarized insights about:

Category performance

Discount behavior

Customer segments (Loyal Customers, Discount Chasers, etc.)



Dependencies:
streamlit – for interactive UI

pandas, numpy – data manipulation

matplotlib, seaborn – visualizations

scikit-learn – PCA, KMeans, scaling

kagglehub (if dataset is downloaded automatically)
