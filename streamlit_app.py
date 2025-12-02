import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Import your analysis functions
from amazon_analysis import (
    load_kaggle_dataset,
    clean_amazon_data,
    q1_category_distribution,
    q2_discount_impact,
    q3_user_behavior_patterns,
    prepare_customer_features,
    perform_kmeans_clustering,
    visualize_clusters_enhanced,
    prepare_data_for_cf,
    build_user_item_matrix,
    compute_user_similarity,
    create_cluster_summary_chart,
    recommend_products
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
    df, _, _ = load_kaggle_dataset(
        "karkavelrajaj/amazon-sales-dataset",
        "amazon.csv",
        "amazon_data.xlsx"
    )
    return clean_amazon_data(df)

@st.cache_data
def prepare_clustering_data(_df):
    """Prepare customer clustering data"""
    customer_features = prepare_customer_features(_df)
    customer_features, X_scaled, scaler = perform_kmeans_clustering(customer_features, n_clusters=4)
    return customer_features, X_scaled

@st.cache_data
def prepare_recommendation_data(_df):
    """Prepare collaborative filtering data"""
    cf_df = prepare_data_for_cf(_df)
    user_item_matrix = build_user_item_matrix(cf_df)
    similarity_df = compute_user_similarity(user_item_matrix)
    return cf_df, user_item_matrix, similarity_df

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
    preview_df['rating'] = preview_df['rating'].map('{:.2f}'.format)
    preview_df['rating_count'] = preview_df['rating_count'].map('{:,}'.format)
    preview_df['discounted_price'] = preview_df['discounted_price'].map('${:,.2f}'.format)
    preview_df['actual_price'] = preview_df['actual_price'].map('${:,.2f}'.format)
    preview_df['discount_percentage'] = preview_df['discount_percentage'].map('{:.0f}%'.format)
    
    st.dataframe(
        preview_df,
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
    cat_summary = pd.DataFrame({
        "Unique Values": [
            df['Main_Category'].nunique(),
            df['user_id'].nunique(),
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
    q1_category_distribution(filtered_df)
    st.pyplot(plt.gcf())
    plt.clf()
    st.markdown("---")
    
    st.subheader("💰 Discount Impact on Ratings")
    q2_discount_impact(filtered_df)
    st.pyplot(plt.gcf())
    plt.clf()
    st.markdown("---")
    
    st.subheader("👥 Customer Behavior & Customer Segments")
    q3_user_behavior_patterns(filtered_df)
    st.pyplot(plt.gcf())
    plt.clf()
    
    # Add Customer Clustering Visualization
    st.subheader("🎯 Customer Segmentation Clusters")
    
    with st.spinner("Preparing customer clustering analysis..."):
        customer_features, X_scaled = prepare_clustering_data(filtered_df)
        
        # Create PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        customer_features['pca1'] = X_pca[:, 0]
        customer_features['pca2'] = X_pca[:, 1]
        
        # Plot clusters
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA scatter plot
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        cluster_names = customer_features['cluster_name'].unique()
        color_map = {name: colors[i % len(colors)] for i, name in enumerate(cluster_names)}
        
        for cluster_name in cluster_names:
            cluster_data = customer_features[customer_features['cluster_name'] == cluster_name]
            if cluster_data.empty:
                continue
            ax1.scatter(
                cluster_data['pca1'], cluster_data['pca2'],
                c=color_map[cluster_name],
                label=f"{cluster_name}",
                alpha=0.6, s=80, edgecolors='white', linewidth=0.5
            )
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
        ax1.set_title('Customer Segmentation (PCA)', fontweight='bold', fontsize=12)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.2)
        
        # Cluster size distribution
        cluster_sizes = customer_features['cluster_name'].value_counts().sort_index()
        bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes.values,
                       color=[colors[i % len(colors)] for i in range(len(cluster_sizes))],
                       edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Cluster', fontweight='bold')
        ax2.set_ylabel('Number of Customers', fontweight='bold')
        ax2.set_title('Cluster Size Distribution', fontweight='bold', fontsize=12)
        ax2.set_xticks(range(len(cluster_sizes)))
        
        cluster_labels = []
        for idx in cluster_sizes.index:
            cluster_data = customer_features[customer_features['cluster_name'] == idx]
            if 'cluster_name' in cluster_data.columns:
                name = cluster_data['cluster_name'].iloc[0]
                cluster_labels.append(name.split()[0])
            else:
                cluster_labels.append(f'C{idx}')
        
        ax2.set_xticklabels(cluster_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.2, axis='y')
        
        for bar, value in zip(bars, cluster_sizes.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cluster_sizes.values)*0.02,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
        
        with st.spinner("Generating cluster heatmap..."):
            heatmap_fig = create_cluster_summary_chart(customer_features)
            st.pyplot(heatmap_fig)


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
        cf_df, user_item_matrix, similarity_df = prepare_recommendation_data(df)
    
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
                    n_recommendations=n_recommendations
                )
                
                # Merge with product details
                rec_with_details = recommendations.merge(
                    df[['product_id', 'product_name', 'rating', 'discounted_price', 'Main_Category']].drop_duplicates('product_id'),
                    on='product_id',
                    how='left'
                )
                
                st.success(f"✅ Found {len(recommendations)} recommendations for user {selected_user}")
                

                # Display recommendations
                st.subheader("🔍Recommended Products")
                
                for idx, row in rec_with_details.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{idx + 1} - {row['product_name'][:100]}{'...' if len(str(row['product_name'])) > 100 else ''}</h4>
                            <p><strong>Product ID:</strong> {row['product_id']}</p>
                            <p><strong>Category:</strong> {row['Main_Category']}</p>
                            <p><strong>Current Rating:</strong> ⭐ {row['rating']:.2f}</p>
                            <p><strong>Price:</strong> ₹{row['discounted_price']:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download recommendations
                st.markdown("---")

                
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
    st.write("- **Discount Chasers** (23.3%): Show highest price sensitivity at 72.1% average discount")
    st.write("- **Casual Shoppers** (10.4%): Demonstrate moderate spending with premium price points")
    

