import os
import zipfile
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')

# Global styling
COLORS = ['#FF6B6B', '#4ECDC4','#06D6A0', '#45B7D1', '#FFA07A', '#F8B739',"#915261", '#A8E6CF',"#334E47"]

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12
})


def load_kaggle_dataset(dataset_id, csv_name):
    """Load Kaggle dataset and perform initial cleaning"""
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    # Download dataset
    dataset_path = kagglehub.dataset_download(dataset_id)
    
    # Handle zip extraction
    if dataset_path.endswith(".zip"):
        extract_path = f"/tmp/{dataset_id.replace('/', '_')}"
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        extract_path = dataset_path
    
    # Load CSV
    file_path = os.path.join(extract_path, csv_name)
    df = pd.read_csv(file_path)
    
    print(f"Loaded {df.shape[0]:,} rows , {df.shape[1]} columns")
    return df


def clean_data(df):
    """Clean and transform the dataset"""
    print("\n" + "="*80)
    print("CLEANING DATA")
    print("="*80)
    
    # Drop unnecessary columns
    df = df.drop(columns=["user_name", "img_link", "product_link"], errors='ignore')
    
    # Clean price columns
    for col in ['discounted_price', 'actual_price']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('₹', '').str.replace(',', ''),
            errors='coerce'
        )
    
    # Clean discount percentage
    df['discount_percentage'] = pd.to_numeric(
        df['discount_percentage'].astype(str).str.replace('%', ''),
        errors='coerce'
    )
    
    # Clean rating and rating_count
    df['rating'] = df['rating'].astype(str).str.replace('|', '4.0')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Clean rating_count
    df['rating_count'] = pd.to_numeric(
        df['rating_count'].astype(str).str.replace(',', ''),
        errors='coerce'
    )

    #Replaced 2 null values in rating_count with median
    df['rating_count'].fillna(df['rating_count'].median(), inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="product_id", keep="first")
    
    # Extract main category
    df['Main_Category'] = df['category'].astype(str).str.split('|').str[0]
    
    # Create new features
    df['discount_amount'] = df['actual_price'] * (df['discount_percentage'] / 100)
    df['review_length'] = df['review_content'].fillna('').str.len()
    df['is_high_rated'] = (df['rating'] >= 4.0).astype(int)
    
    # Save cleaned data
    df.to_excel("amazon_cleaned.xlsx", index=False)
    print(f"Cleaned dataset: {df.shape[0]:,} rows")
    print(f"Saved to: amazon_cleaned.xlsx")
    
    return df




def explore_dataframe(df):
    """Comprehensive DataFrame exploration"""

    pd.set_option('display.max_columns', None)
    
    print("\n" + "="*80)
    print("DATAFRAME EXPLORATION")
    print("="*80)
    
    #shape
    print(f"\nSHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    
    #data type
    print("\nDATA TYPES:")
    print(df.dtypes)
    
    #check if there missing value
    print("\nMISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0])

    #check if there missing value
    print(f"\nDUPLICATED ROWS: {df.duplicated().sum()}")

    #print statistics 
    print("\nSTATISTICS:")
    print(df.describe(include='all'))

    #print first 5 row
    print("\nFIRST 5 ROWS:")
    print(df.head())
    return df


def generate_summary(df):
    """Generate executive summary"""
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    print(f"Total Products: {len(df):,}")
    print(f"Total Reviews: {int(df['rating_count'].sum()):,}")
    print(f"Average Rating: {df['rating'].mean():.2f}")
    print(f"Average Discount: {df['discount_percentage'].mean():.1f}%")
    print(f"Average Price: ₹{df['discounted_price'].mean():.2f}")
    print(f"Total Categories: {df['Main_Category'].nunique()}")
    print(f"Total Users: {df['user_id'].str.split(',').explode().str.strip().nunique()}")
    
    high_rated_pct = (df['rating'] >= 4.0).mean() * 100
    print(f"\nHigh-rated products (≥4.0): {(df['rating'] >= 4.0).sum():,} ({high_rated_pct:.1f}%)")
    print(f"Heavy discounts (>50%): {(df['discount_percentage'] > 50).sum():,}")
    print(f"Most popular category: {df['Main_Category'].mode()[0]}")


def analyze_categories(df):
    """Analyze product distribution across categories"""
    print("\n" + "="*80)
    print("CATEGORY ANALYSIS")
    print("="*80)
    
    category_stats = df.groupby('Main_Category').agg({
        'product_id': 'count',
        'rating_count': 'sum',
        'rating': 'mean'
    }).round(2)
    category_stats.columns = ['Product_Count', 'Total_Reviews', 'Avg_Rating']
    category_stats = category_stats.sort_values('Product_Count', ascending=False)
    
    print("\nThe data:")
    print(category_stats.head(10))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color mapping
    extended_palette = (COLORS * (len(category_stats) // len(COLORS) + 1))[:len(category_stats)]
    color_map = {cat: extended_palette[i] for i, cat in enumerate(category_stats.index)}
    # Bar chart
    bars = axes[0].barh(
        category_stats.index[:15],
        category_stats['Product_Count'][:15],
        color=[color_map[cat] for cat in category_stats.index[:15]]
    )
    axes[0].set_xlabel('Number of Products')

    axes[0].set_title('Number of Products per Category')
    axes[0].invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        axes[0].text(
            width, bar.get_y() + bar.get_height()/2,
            f'{int(width)}',
            ha='left', va='center', fontsize=9
        )
    
    # Scatter plot
    category_stats_sorted = category_stats.sort_values('Total_Reviews', ascending=True)
    axes[1].scatter(
        category_stats_sorted['Total_Reviews'],
        category_stats_sorted['Avg_Rating'],
        s=category_stats_sorted['Product_Count']*6,
        c=[color_map[cat] for cat in category_stats_sorted.index],
        alpha=0.6
    )
    axes[1].set_xlabel('Total Reviews')
    axes[1].set_ylabel('Average Rating')
    axes[1].set_title('Total Reviews vs Average Rating by Category')
    
    plt.tight_layout()
    plt.show(block=False)
    
    return category_stats

def format_large_numbers(x):
    """Format large numbers for display"""
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    elif x >= 1_000:
        return f"{x/1_000:.2f}K"
    else:
        return f"{x:.0f}"
    
def analyze_discounts(df):
    """Analyze discount impact on ratings and reviews"""
    print("\n" + "="*80)
    print("DISCOUNT ANALYSIS")
    print("="*80)
    
    df['discount_bin'] = pd.cut(
        df['discount_percentage'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    )
    
    discount_stats = df.groupby('discount_bin').agg({
        'rating': 'mean',
        'rating_count': 'sum'
    }).round(2)
    
    print("\nDiscount Impact:")
    print(discount_stats)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average rating
    rating_sorted = discount_stats.sort_values('rating', ascending=False)
    axes[0].bar(rating_sorted.index, rating_sorted['rating'], color=COLORS[0])
    axes[0].set_title('Average Rating by Discount Range')
    axes[0].set_ylabel('Average Rating')
    for i, v in enumerate(rating_sorted['rating']):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    # Total reviews
    reviews_sorted = discount_stats.sort_values('rating_count', ascending=False)
    reviews_sorted['rating_count_fmt'] = reviews_sorted['rating_count'].apply(format_large_numbers)
    axes[1].bar(reviews_sorted.index, reviews_sorted['rating_count'], color=COLORS[1])
    axes[1].set_title('Total Reviews by Discount Range')
    axes[1].set_ylabel('Total Reviews')
    discount_stats['rating_count_fmt'] = discount_stats['rating_count'].apply(format_large_numbers)
    for i, v in enumerate(reviews_sorted['rating_count']):
        offset = v * 0.01 
        axes[1].text(i, v + offset, reviews_sorted['rating_count_fmt'].iloc[i],
                     ha='center', va='bottom')
    plt.tight_layout()
    plt.show(block=False)
    
    return discount_stats


def analyze_user_behavior(df):
    """Analyze user review patterns"""
    print("\n" + "="*80)
    print("USER BEHAVIOR ANALYSIS")
    print("="*80)
    
    # Expand user_ids
    df_expanded = df.assign(user_id=df['user_id'].str.split(',')).explode('user_id')
    df_expanded['user_id'] = df_expanded['user_id'].str.strip()
    
    # User statistics
    user_stats = df_expanded.groupby('user_id').agg(
        Review_Count=('review_id', 'count'),
        Avg_Rating=('rating', 'mean')
    ).reset_index()
    
    # Segment users
    user_stats['Segment'] = pd.cut(
        user_stats['Review_Count'],
        bins=[0, 1, 4, 9, np.inf],
        labels=['One-time (1)', 'Occasional (2-4)', 'Active (5-9)', 'Power (10+)']
    )
    
    # Segment analysis
    segment_stats = user_stats.groupby('Segment').agg(
        User_Count=('user_id', 'count'),
        Avg_Reviews=('Review_Count', 'mean'),
        Avg_Rating=('Avg_Rating', 'mean')
    ).round(2)
    
    print("\nUser Segments:")
    print(segment_stats)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    # Pie chart
    axes[0].pie(
        segment_stats['User_Count'],
        labels=[f"{seg}\n({count:,})" for seg, count in zip(segment_stats.index, segment_stats['User_Count'])],
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '', 
        colors=COLORS[:4],
        startangle=90,
        pctdistance=0.7,  
        labeldistance=1.1 
    )
    axes[0].set_title('User Distribution by Segment')
    
    # Box plot
    segment_data = [user_stats[user_stats['Segment'] == seg]['Avg_Rating'].values
                   for seg in segment_stats.index]
    box = axes[1].boxplot(segment_data, labels=segment_stats.index, patch_artist=True)
    for patch, color in zip(box['boxes'], COLORS[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Average Rating')
    axes[1].set_title('Rating Distribution by Segment')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show(block=False)
    
    return user_stats




def recommend_products(user_id, user_item_matrix, similarity_df, products_df, n_recommendations=5):
    """Generate product recommendations for a user"""
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()
    
    # Find similar users
    similarities = similarity_df[user_id].sort_values(ascending=False)
    similar_users = similarities[(similarities.index != user_id) & (similarities > 0)].head(10)
    
    # Products already rated
    if similar_users.empty:
        print("\n No similar users found")
        user_rated = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
        popular = user_item_matrix.sum().sort_values(ascending=False)
        popular = popular[~popular.index.isin(user_rated)]
        
        recommendations = []
        for product_id in popular.index[:n_recommendations]:
            product_info = products_df[products_df['product_id'] == product_id].iloc[0]
            recommendations.append({
                'product_id': product_id,
                'product_name': product_info['product_name'],
                'rating': product_info['rating']
            })
        return pd.DataFrame(recommendations)
    
    user_rated = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
    
    # Calculate recommendation scores
    scores = {}
    for similar_user, sim_score in similar_users.items():
        similar_ratings = user_item_matrix.loc[similar_user]
        for product, rating in similar_ratings[similar_ratings > 0].items():
            if product not in user_rated:
                if product not in scores:
                    scores[product] = {'score': 0, 'count': 0}
                scores[product]['score'] += sim_score * rating
                scores[product]['count'] += 1
    
    if not scores:
        user_rated = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
        popular = user_item_matrix.sum().sort_values(ascending=False)
        popular = popular[~popular.index.isin(user_rated)]
        
        recommendations = []
        for product_id in popular.index[:n_recommendations]:
            product_info = products_df[products_df['product_id'] == product_id].iloc[0]
            recommendations.append({
                'product_id': product_id,
                'product_name': product_info['product_name'],
                'rating': product_info['rating']
            })
        return pd.DataFrame(recommendations)
    
    # Build recommendations from scored products
    recommendations = []
    for product, data in sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True):
        product_info = products_df[products_df['product_id'] == product].iloc[0]
        recommendations.append({
            'product_id': product,
            'product_name': product_info['product_name'],
            'rating': product_info['rating']
        })
    
    # If we have fewer than n_recommendations, fill with popular products
    if len(recommendations) < n_recommendations:
        popular = user_item_matrix.sum().sort_values(ascending=False)
        popular = popular[~popular.index.isin(user_rated)]
        
        # Add popular products not already in recommendations
        for product_id in popular.index:
            if product_id not in [rec['product_id'] for rec in recommendations]:
                product_info = products_df[products_df['product_id'] == product_id].iloc[0]
                recommendations.append({
                    'product_id': product_id,
                    'product_name': product_info['product_name'],
                    'rating': product_info['rating']
                })
                if len(recommendations) >= n_recommendations:
                    break
    
    return pd.DataFrame(recommendations[:n_recommendations])



def build_recommender_system(df):
    """Build collaborative filtering recommender"""
    print("\n" + "="*80)
    print("COLLABORATIVE FILTERING RECOMMENDER")
    print("="*80)
    
    # Prepare data
    cf_df = df.assign(user_id=df['user_id'].str.split(',')).explode('user_id')
    cf_df['user_id'] = cf_df['user_id'].str.strip()
    cf_df = cf_df[['user_id', 'product_id', 'rating']].dropna()
    
    # Build user-item matrix
    user_item_matrix = cf_df.pivot_table(
        index='user_id',
        columns='product_id',
        values='rating',
        fill_value=0
    )
    
    # Compute similarity
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    sparsity = (user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100
    print(f"Matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {sparsity:.2f}%")
    
    return user_item_matrix, similarity_df








def cluster_customers(df):
    """Perform customer segmentation into 4 distinct clusters"""
    print("\n" + "="*80)
    print("CUSTOMER SEGMENTATION")
    print("="*80)
    
    # Expand user_ids and aggregate customer features
    df_expanded = df.assign(user_id=df['user_id'].str.split(',')).explode('user_id')
    df_expanded['user_id'] = df_expanded['user_id'].str.strip()
    
    customer_features = df_expanded.groupby('user_id').agg({
        'review_id': 'count',
        'rating': ['mean', 'std'],
        'discounted_price': ['mean', 'sum'],
        'discount_percentage': 'mean',
        'review_content': lambda x: x.str.len().mean()
    }).reset_index()
    
    customer_features.columns = ['user_id', 'review_count', 'avg_rating', 'rating_std',
                                  'avg_price', 'total_spending', 'avg_discount', 'avg_review_length']
    customer_features.fillna(0, inplace=True)
    
    # Key engineered features
    customer_features['log_spending'] = np.log1p(customer_features['total_spending'])
    customer_features['spending_per_review'] = customer_features['total_spending'] / customer_features['review_count']
    customer_features['rating_consistency'] = 1 / (1 + customer_features['rating_std'])
    customer_features['loyalty_index'] = (customer_features['review_count'] * 
                                           customer_features['rating_consistency'] * 
                                           (1 + customer_features['avg_rating']))
    
    # Print customer distribution
    print(f"\nTotal unique customers: {len(customer_features):,}")
    one_time_count = (customer_features['review_count'] == 1).sum()
    repeat_count = (customer_features['review_count'] > 1).sum()
    print(f"One-time buyers: {one_time_count:,} ({one_time_count / len(customer_features) * 100:.1f}%)")
    print(f"Repeat buyers: {repeat_count:,} ({repeat_count / len(customer_features) * 100:.1f}%)")
    
    # Separate one-time buyers
    one_time_mask = customer_features['review_count'] == 1
    repeat_mask = customer_features['review_count'] > 1
    
    customer_features['cluster_name'] = 'One-Time Buyers'
    customer_features['cluster'] = 0
    customer_features.loc[one_time_mask, 'cluster'] = 0
    
    # Cluster repeat buyers into 3 segments
    if repeat_count > 0:
        feature_cols = ['review_count', 'avg_rating', 'rating_consistency', 
                        'log_spending', 'spending_per_review', 'avg_discount', 
                        'loyalty_index']
        
        X_repeat = customer_features.loc[repeat_mask, feature_cols].copy().fillna(0)
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        X_repeat_scaled = scaler.fit_transform(X_repeat)
        
        pca_reducer = PCA(n_components=0.95)
        X_repeat_pca = pca_reducer.fit_transform(X_repeat_scaled)
        
        print(f"\nPCA reduced to {X_repeat_pca.shape[1]} components (variance: {pca_reducer.explained_variance_ratio_.sum():.1%})")
        
        # K-Means with k=3 for repeat buyers
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=100, max_iter=1000)
        labels = kmeans.fit_predict(X_repeat_pca)
        
        customer_features.loc[repeat_mask, 'cluster'] = labels + 1  # Offset by 1 (0 is One-Time Buyers)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_repeat_pca, labels)
        davies_bouldin = davies_bouldin_score(X_repeat_pca, labels)
        
        print(f"\nClustering Metrics (Repeat Buyers):")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        # Analyze clusters and assign names
        repeat_profiles = customer_features[repeat_mask].groupby('cluster').agg({
            'review_count': 'mean',
            'avg_rating': 'mean',
            'total_spending': 'mean',
            'avg_discount': 'mean',
            'loyalty_index': 'mean'
        })
        
        # Map clusters to names based on characteristics
        cluster_mapping = {0: 'One-Time Buyers'}
        
        for cluster_id in repeat_profiles.index:
            profile = repeat_profiles.loc[cluster_id]
            
            # Loyal Customers: High loyalty index and review count
            if profile['loyalty_index'] > repeat_profiles['loyalty_index'].quantile(0.70):
                cluster_mapping[cluster_id] = 'Loyal Customers'
            # Discount Seekers: High discount percentage
            elif profile['avg_discount'] > repeat_profiles['avg_discount'].quantile(0.60):
                cluster_mapping[cluster_id] = 'Discount Seekers'
            # Casual Shoppers: Default
            else:
                cluster_mapping[cluster_id] = 'Casual Shoppers'
        
        # Apply mapping
        customer_features['cluster_name'] = customer_features['cluster'].map(cluster_mapping)
        
        # Prepare data for visualization
        X_all = customer_features[feature_cols].copy().fillna(0)
        scaler_all = StandardScaler()
        X_scaled = scaler_all.fit_transform(X_all)
    else:
        X_scaled = None
    
    # Print cluster distribution
    print("\nCluster Distribution:")
    cluster_counts = customer_features['cluster_name'].value_counts()
    for cluster_name in ['One-Time Buyers', 'Casual Shoppers', 'Discount Seekers', 'Loyal Customers']:
        if cluster_name in cluster_counts.index:
            count = cluster_counts[cluster_name]
            pct = (count / len(customer_features)) * 100
            print(f"  {cluster_name}: {count:,} ({pct:.1f}%)")
    
    # Visualize
    if X_scaled is not None:
        visualize_clusters(customer_features, X_scaled)
        create_cluster_heatmap(customer_features)
    
    # Save results
    customer_features.to_excel("customer_clusters.xlsx", index=False)
    print("\nSaved to: customer_clusters.xlsx")
    
    return customer_features,X_scaled
def visualize_clusters(customer_features, X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    customer_features['pca1'] = X_pca[:, 0]
    customer_features['pca2'] = X_pca[:, 1]

    COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    fig = plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(1, 3, 1)
    for i, cluster_name in enumerate(['One-Time Buyers', 'Casual Shoppers', 'Discount Seekers', 'Loyal Customers']):
        if cluster_name in customer_features['cluster_name'].unique():
            cluster_data = customer_features[customer_features['cluster_name'] == cluster_name]
            ax1.scatter(cluster_data['pca1'], cluster_data['pca2'], c=COLORS[i], label=cluster_name, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Customer Segmentation - PCA (PC1 vs PC2)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Pie chart
    ax3 = plt.subplot(1, 3, 3)
    cluster_sizes = customer_features['cluster_name'].value_counts()
    def autopct_format(pct): return f'{pct:.1f}%' if pct >= 6 else ''
    wedges, texts, autotexts = ax3.pie(
        cluster_sizes.values,
        labels=None,
        colors=[COLORS[['One-Time Buyers', 'Casual Shoppers', 'Discount Seekers', 'Loyal Customers'].index(name)] for name in cluster_sizes.index],
        autopct=autopct_format,
        startangle=90,
        explode=[0.05]*len(cluster_sizes),
        pctdistance=0.5
    )
    legend_labels = [f"{name}: {count}" for name, count in zip(cluster_sizes.index, cluster_sizes.values)]
    ax3.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax3.set_title('Cluster Distribution')

    plt.tight_layout()
    return fig

def create_cluster_heatmap(customer_features):
    cluster_profiles = customer_features.groupby('cluster_name').agg({
        'review_count': 'mean',
        'avg_rating': 'mean',
        'avg_price': 'mean',
        'total_spending': 'mean',
        'spending_per_review': 'mean',
        'avg_discount': 'mean'
    }).round(2)
    cluster_profiles.columns = ['Avg Reviews', 'Avg Rating', 'Avg Price (₹)', 'Total Spending (₹)', 'Spending/Review (₹)', 'Avg Discount %']

    cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_profiles_norm.T, annot=cluster_profiles.T.values, fmt='.1f', cmap='YlOrRd', linewidths=2, linecolor='white', ax=ax)
    ax.set_title('Cluster Profile Heatmap\n(Actual values with relative intensity)')
    ax.set_xlabel('Customer Segments')
    ax.set_ylabel('Behavioral Metrics')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Load, explore and clean data
    df = load_kaggle_dataset("karkavelrajaj/amazon-sales-dataset", "amazon.csv")
    df = explore_dataframe(df)
    df = clean_data(df) 
    df = explore_dataframe(df)

    # Generate summary
    generate_summary(df)

    #----------------------------------------

    # Run analyses
    analyze_categories(df)
    analyze_discounts(df)
    analyze_user_behavior(df)

    #----------------------------------------

    # Build recommender system
    user_item_matrix, similarity_df = build_recommender_system(df)
    
    # Example recommendation
    user_id = "AHFQGP45QKIEFKYOCYUH4DP63XGQ"
    recommendations = recommend_products(user_id, user_item_matrix, similarity_df,df)
    if not recommendations.empty:
        print(f"\nTop 5 recommendations for user {user_id}:")
        print(recommendations[['product_id', 'product_name', 'rating']])
    
    #----------------------------------------

    #Customer segmentation
    cluster_customers(df)
    
    create_cluster_heatmap
    visualize_clusters


    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    input("\nPress Enter to close all plots and exit...")
    plt.close('all')
