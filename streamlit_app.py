import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Import your analysis functions - updated to match actual file
from amazon_analysis import (
    load_kaggle_dataset,
    clean_data,
    analyze_categories,
    analyze_discounts,
    analyze_user_behavior,
    build_recommender_system,
    recommend_products,
    visualize_clusters,
    create_cluster_heatmap,
    cluster_customers
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Amazon Sales Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    color: #FF9900;
    padding: 10px 0 20px 0;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #333;
    margin-top: 20px;
}
.insight-box {
    background:#e8f4f8;
    padding:1rem;
    margin-top:20px;
    border-left:4px solid #1f77b4;
    border-radius:5px;
}
.recommendation-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px 0;
    border-left: 4px solid #FF9900;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load and clean data
# -----------------------------
@st.cache_data
def load_and_clean_data():
    """Load and clean the dataset"""
    df = load_kaggle_dataset("karkavelrajaj/amazon-sales-dataset", "amazon.csv")
    df = clean_data(df)
    return df

@st.cache_data
def prepare_recommendation_data(_df):
    """Prepare collaborative filtering data"""
    user_item_matrix, similarity_df = build_recommender_system(_df)
    return user_item_matrix, similarity_df

# Load data
with st.spinner("Loading data..."):
    df = load_and_clean_data()

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=180)
page = st.sidebar.radio(
    "Pages:",
    [
        "🧾 Dataset Description",
        "🔎 Data Preview",
        "📈 Summary Statistics",
        "📊 Interactive Visualizations",
        "🎯 Product Recommendations",
        "💡 Insight Section"
    ]
)

# -----------------------------
# Page: Dataset Description
# -----------------------------
if page == "🧾 Dataset Description": 
    st.markdown(
        '<h3 style="text-align: center;">🧾 Dataset Description</h3>',
        unsafe_allow_html=True
    )

    st.write("""
    This dataset was obtained from **Kaggle** and contains Amazon product information 
    including prices, discounts, ratings, reviews, and user interactions.  
    The data represents products scraped in **January 2023** from the official Amazon website.
    """)

    st.markdown("""
    ### Features Included
    - **product_id** – Unique product identifier  
    - **product_name** – Name of the product  
    - **category** – Product category  
    - **discounted_price** – Current selling price  
    - **actual_price** – Original price before discount  
    - **discount_percentage** – Discount applied  
    - **rating** – Average customer rating  
    - **rating_count** – Number of customer ratings  
    - **about_product** – Product description  
    - **user_id** – Reviewer's user ID  
    - **user_name** – Reviewer's name  
    - **review_id** – Unique review ID  
    - **review_title** – Short review headline  
    - **review_content** – Full customer review  
    - **img_link** – Product image link  
    - **product_link** – Official Amazon product page  
    """)

# -----------------------------
# Page: Data Preview
# -----------------------------
elif page == "🔎 Data Preview":
    st.markdown(
        '<h3 style="text-align: center;">🔎 Data Preview</h3>',
        unsafe_allow_html=True
    )
    
    # Select relevant columns
    preview_df = df[['product_name', 'Main_Category', 'rating', 'rating_count', 
                     'discounted_price', 'actual_price', 'discount_percentage']].head(100)
    
    # Format numeric columns
    preview_df_display = preview_df.copy()
    preview_df_display['rating'] = preview_df_display['rating'].map('{:.2f}'.format)
    preview_df_display['rating_count'] = preview_df_display['rating_count'].map('{:,.0f}'.format)
    preview_df_display['discounted_price'] = preview_df_display['discounted_price'].map('₹{:,.2f}'.format)
    preview_df_display['actual_price'] = preview_df_display['actual_price'].map('₹{:,.2f}'.format)
    preview_df_display['discount_percentage'] = preview_df_display['discount_percentage'].map('{:.0f}%'.format)
    
    st.dataframe(
        preview_df_display,
        height=400,
        use_container_width=True
    )

# -----------------------------
# Page: Summary Statistics
# -----------------------------
elif page == "📈 Summary Statistics":
    st.markdown(
        '<h3 style="text-align: center;">📈 Summary Statistics</h3>',
        unsafe_allow_html=True
    )
    
    st.subheader("🔢 Numerical Features")
    num_cols = ['rating', 'rating_count', 'discounted_price', 'actual_price', 'discount_percentage']
    
    num_stats = df[num_cols].describe().T.round(2)
    num_stats['median'] = df[num_cols].median().round(2)
    num_stats['missing_values'] = df[num_cols].isnull().sum()
    num_stats['skewness'] = df[num_cols].skew().round(2)
    
    st.dataframe(num_stats, use_container_width=True)

    st.subheader("📊 Dataset Overview")
    
    # Expand user_ids for unique count
    df_expanded = df.assign(user_id=df['user_id'].str.split(',')).explode('user_id')
    df_expanded['user_id'] = df_expanded['user_id'].str.strip()
    
    cat_summary = pd.DataFrame({
        "Unique Values": [
            df['Main_Category'].nunique(),
            df_expanded['user_id'].nunique(),
            df['product_name'].nunique(),
            df['review_id'].nunique()
        ]
    }, index=["Categories", "Users", "Products", "Reviews"])
    
    st.dataframe(cat_summary, use_container_width=True)

# -----------------------------
# Page: Interactive Visualizations (with filters)
# -----------------------------
elif page == "📊 Interactive Visualizations":
    st.markdown(
        '<h3 style="text-align: center;">📊 Interactive Visualizations</h3>',
        unsafe_allow_html=True
    )
    
    # Filters
    st.markdown("### 🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        categories = sorted(df['Main_Category'].unique())
        selected_categories = st.multiselect("Category", options=categories, default=[])
    with col2:
        rating_filter = st.slider("Minimum Rating", 0.0, 5.0, 0.0, step=0.5)
    with col3:
        price_min, price_max = float(df['discounted_price'].min()), float(df['discounted_price'].max())
        price_range = st.slider("Price Range", price_min, price_max, (price_min, price_max))
    with col4:
        discount_range = st.slider("Discount %", 0, 100, (0, 100))
    
    # Apply filters
    filtered_df = df.copy()
    if selected_categories:
        filtered_df = filtered_df[filtered_df['Main_Category'].isin(selected_categories)]
    filtered_df = filtered_df[
        (filtered_df['rating'] >= rating_filter) &
        (filtered_df['discounted_price'].between(*price_range)) &
        (filtered_df['discount_percentage'].between(*discount_range))
    ]

    # Key Metrics
    st.markdown("### 📌 Key Metrics")
    kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
    with kcol1:
        st.metric(
            "Total Products",
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if selected_categories else None
        )
    with kcol2:
        st.metric(
            "Avg Rating",
            f"{filtered_df['rating'].mean():.2f}",
            delta=f"{filtered_df['rating'].mean() - df['rating'].mean():.2f}" if selected_categories else None
        )
    with kcol3:
        st.metric(
            "Avg Discount",
            f"{filtered_df['discount_percentage'].mean():.1f}%",
            delta=f"{filtered_df['discount_percentage'].mean() - df['discount_percentage'].mean():.1f}%" if selected_categories else None
        )
    with kcol4:
        st.metric(
            "Avg Price",
            f"₹{filtered_df['discounted_price'].mean():,.0f}",
            delta=f"₹{filtered_df['discounted_price'].mean() - df['discounted_price'].mean():,.0f}" if selected_categories else None
        )
    with kcol5:
        st.metric(
            "Total Reviews",
            f"{filtered_df['rating_count'].sum():,.0f}"
        )

    # Visualizations
    st.subheader("📦 Category Distribution")
    with st.spinner("Generating category analysis..."):
        analyze_categories(filtered_df)
        for fig_num in plt.get_fignums():
            st.pyplot(plt.figure(fig_num))
        plt.close('all')
    
    st.markdown("---")
    
    st.subheader("💰 Discount Impact on Ratings")
    with st.spinner("Analyzing discount impact..."):
        analyze_discounts(filtered_df)
        for fig_num in plt.get_fignums():
            st.pyplot(plt.figure(fig_num))
        plt.close('all')
    
    st.markdown("---")
    
    st.subheader("👥 User Behavior Patterns")
    with st.spinner("Analyzing user behavior..."):
        analyze_user_behavior(filtered_df)
        for fig_num in plt.get_fignums():
            st.pyplot(plt.figure(fig_num))
        plt.close('all')
    
    st.markdown("---")
    
    # Customer Clustering Visualization 
    st.subheader("🎯 Customer Segmentation Clusters")
    with st.spinner("Customer Segmentation Clusters..."):
        customer_features, X_scaled = cluster_customers(filtered_df)

        if X_scaled is not None:
            fig_clusters = visualize_clusters(customer_features, X_scaled)
            st.pyplot(fig_clusters)

            fig_heatmap = create_cluster_heatmap(customer_features)
            st.pyplot(fig_heatmap)
        else:
            st.warning("Not enough repeat buyers to generate clusters.")

# -----------------------------
# Page: Product Recommendations
# -----------------------------
elif page == "🎯 Product Recommendations":
    st.markdown(
        '<h3 style="text-align: center;">🎯 Product Recommendations</h3>',
        unsafe_allow_html=True
    )
    
    st.write("""
    This page provides personalized product recommendations using **Collaborative Filtering**. 
    Select a user ID to see recommended products based on similar users' preferences.
    """)
    
    with st.spinner("Preparing recommendation system..."):
        user_item_matrix, similarity_df = prepare_recommendation_data(df)
    
    # User selection
    st.subheader("👤 Select User")
    available_users = sorted(user_item_matrix.index.tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_user = st.selectbox(
            "Choose a User ID:",
            options=available_users,
            index=0
        )
    with col2:
        n_recommendations = st.number_input(
            "Number of Recommendations:",
            min_value=1,
            max_value=20,
            value=5
        )
    
    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                # Get recommendations
                recommendations = recommend_products(
                    selected_user, 
                    user_item_matrix, 
                    similarity_df,
                    df,
                    n_recommendations=n_recommendations
                )
                
                if recommendations.empty:
                    st.warning(f"⚠️ No recommendations available for user {selected_user}")
                else:
                    st.success(f"✅ Found {len(recommendations)} recommendations for user {selected_user}")
                    
                    # Display recommendations
                    st.subheader("🔍 Recommended Products")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>#{idx + 1} - {row['product_name'][:100]}{'...' if len(str(row['product_name'])) > 100 else ''}</h4>
                                <p><strong>Product ID:</strong> {row['product_id']}</p>
                                <p><strong>Rating:</strong> ⭐ {row['rating']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")

# -----------------------------
# Page: Insight Section
# -----------------------------
elif page == "💡 Insight Section":
    st.markdown(
        '<h3 style="text-align: center;">💡 Insight Section</h3>',
        unsafe_allow_html=True
    )
    
    st.subheader("📦 Category Insights")
    st.write("- Big categories (Electronics, Home & Kitchen, Computers) get most reviews but have average ratings, keeping the overall weighted rating around **4.09**.")
    st.write("- Small categories (Office Products, Toys & Games, Home Improvement) have fewer products but higher ratings, above **4.25**.")
    st.write("- Car & Motorbike and Musical Instruments have low reviews and lower ratings, performing below the overall average.")
    
    st.write("---")

    st.subheader("💰 Discount Insights")
    st.write("- Higher ratings are seen at lower discounts.")
    st.write("- Most reviews come from mid-range discounts (40–80%).")
    st.write("- Very high discounts (80–100%) have the lowest rating and fewest reviews.")
    
    st.write("---")

    st.subheader("👥 Customer Behavior & Customer Segments")
    st.write("- Most reviews come from **one-time reviewers** (62.4% of customers).")
    st.write("- Frequent reviewers rate slightly higher.")
    st.write("- **Loyal Customers** (3.9%): Spend **₹29,084** on average (25x more than one-time buyers)")
    st.write("- **One-Time Buyers** (62.4%): Represent the largest segment with low engagement")
    st.write("- **Discount Seekers** (23.3%): Show highest price sensitivity at 72.1% average discount")
    st.write("- **Casual Shoppers** (10.4%): Demonstrate moderate spending with premium price points")
