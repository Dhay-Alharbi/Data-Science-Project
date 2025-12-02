import os
import zipfile
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import openpyxl

# Global color palette for consistent styling
GLOBAL_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F8B739', '#A8E6CF']
CATEGORY_PALETTE = sns.color_palette('RdYlBu_r', n_colors=20)


warnings.filterwarnings('ignore')


def load_kaggle_dataset(dataset_id, csv_name, excel_name="output.xlsx"):
    """Download and load Kaggle dataset"""
    print("Loading Kaggle dataset...")
    
    dataset_path = kagglehub.dataset_download(dataset_id)
    print(f"Dataset path: {dataset_path}")
    
    # Handle zip files
    if dataset_path.endswith(".zip"):
        extract_path = f"/tmp/{dataset_id.replace('/', '_')}"
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted to: {extract_path}")
    else:
        extract_path = dataset_path
    
    print(f"Files: {os.listdir(extract_path)}")
    
    # Load CSV
    file_path = os.path.join(extract_path, csv_name)
    df = pd.read_csv(file_path)
    
    # Save Excel
    excel_path = os.path.join(os.getcwd(), excel_name)
    df.to_excel(excel_path, index=False, engine="openpyxl")

    print(f"Excel saved: {excel_path}")
    
    return df, extract_path, excel_path


def explore_dataframe(df):
    """Comprehensive DataFrame exploration"""
    pd.set_option('display.max_columns', None)
    
    print("\n" + "="*80)
    print("DATAFRAME EXPLORATION")
    print("="*80)
    
    print(f"\nSHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\nDATA TYPES:")
    print(df.dtypes)
    
    print("\nMISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0])
    
    print(f"\nDUPLICATED ROWS: {df.duplicated().sum()}")
    
    print("\nSTATISTICS:")
    print(df.describe(include='all'))
    
    print("\nFIRST 5 ROWS:")
    print(df.head())
    return df


def clean_amazon_data(df):
    """Clean and preprocess Amazon dataset"""
    print("\nCleaning Amazon data...")

    # Drop unnecessary columns
    columns_to_drop = ["user_name", "img_link", "product_link"]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    
    # Clean price columns
    price_cols = ['discounted_price', 'actual_price']
    for col in price_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('₹', '').str.replace(',', ''),
            errors='coerce'
        )
    
    # Clean discount percentage
    df['discount_percentage'] = pd.to_numeric(
        df['discount_percentage'].astype(str).str.replace('%', ''),
        errors='coerce'
    )
    
    # Clean rating
    df['rating'] = df['rating'].astype(str).str.replace('|', '4.0')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Clean rating_count
    df['rating_count'] = pd.to_numeric(
        df['rating_count'].astype(str).str.replace(',', ''),
        errors='coerce'
    )
    df['rating_count'].fillna(df['rating_count'].median(), inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="product_id", keep="first")
    
    # Extract main category
    if 'category' in df.columns:
        df['Main_Category'] = df['category'].astype(str).str.split('|').str[0]
    
    # Create derived features
    df['discount_amount'] = df['actual_price'] * (df['discount_percentage'] / 100)
    df['review_length'] = df['review_content'].fillna('').str.len()
    df['is_high_rated'] = (df['rating'] >= 4.0).astype(int)
    
    # Save cleaned data
    excel_path = os.path.join(os.getcwd(), "amazon_cleaned.xlsx")
    df.to_excel(excel_path, index=False,engine="openpyxl")
    print(f"Cleaned data saved: {excel_path}")
    
    return df


def format_large_numbers(x):
    """Format large numbers for display"""
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    elif x >= 1_000:
        return f"{x/1_000:.2f}K"
    else:
        return f"{x:.0f}"


def q1_category_distribution(df):
    """Analyze product distribution across categories"""
    print("\nAnalyzing category distribution...")
    
    category_stats = df.groupby('Main_Category').agg({
        'product_id': 'count',
        'rating_count': 'sum',
        'rating': 'mean'
    }).round(2)
    category_stats.columns = ['Product_Count', 'Total_Reviews', 'Avg_Rating']
    category_stats = category_stats.sort_values('Product_Count', ascending=False)
    
    print("\nThe head:")
    print(category_stats.head(10))
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color mapping
    extended_palette = (GLOBAL_COLORS * (len(category_stats) // len(GLOBAL_COLORS) + 1))[:len(category_stats)]
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
        s=category_stats_sorted['Product_Count']*3,
        c=[color_map[cat] for cat in category_stats_sorted.index],
        alpha=0.6
    )
    axes[1].set_xlabel('Total Reviews')
    axes[1].set_ylabel('Average Rating')
    axes[1].set_title('Reviews vs Ratings (Bubble Size = Product Count)')
    
    plt.tight_layout()
    plt.show(block=False)
    
    return category_stats


def q2_discount_impact(df): 
    df['discount_bin'] = pd.cut(df['discount_percentage'],
                                bins=[0, 20, 40, 60, 80, 100],
                                labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    
    discount_stats = df.groupby('discount_bin').agg({
        'rating':'mean',
        'rating_count':'sum'
    }).round(2)

    # Sort by rating_count for total reviews chart
    discount_stats_reviews = discount_stats.sort_values('rating_count', ascending=False)
    discount_stats_reviews['rating_count_fmt'] = discount_stats_reviews['rating_count'].apply(format_large_numbers)

    # Sort by rating for average rating chart
    discount_stats_rating = discount_stats.sort_values('rating', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Average Rating by Discount Bin (sorted by rating descending)
    axes[0].bar(discount_stats_rating.index, discount_stats_rating['rating'], color='#FF6B6B')
    axes[0].set_title('Average Rating by Discount Bin')
    axes[0].set_ylabel('Average Rating')
    for i, v in enumerate(discount_stats_rating['rating']):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha='center')

    # Total Reviews by Discount Bin (sorted by rating_count descending)
    axes[1].bar(discount_stats_reviews.index, discount_stats_reviews['rating_count'], color='#4ECDC4')
    axes[1].set_title('Total Reviews by Discount Bin')
    axes[1].set_ylabel('Total Reviews')
    for i, v in enumerate(discount_stats_reviews['rating_count']):
        axes[1].text(i, v + max(discount_stats_reviews['rating_count'])*0.01,
                     discount_stats_reviews['rating_count_fmt'].iloc[i], ha='center')

    plt.tight_layout()
    plt.show(block=False)
    
    return discount_stats

def q3_user_behavior_patterns(df):
    """Analyze user review patterns and behaviors"""
    print("\nAnalyzing user behavior patterns...")
    
    # Expand multiple user_ids per review
    df_expanded = df.assign(user_id=df['user_id'].str.split(',')).explode('user_id')
    df_expanded['user_id'] = df_expanded['user_id'].str.strip()
    
    # Compute user statistics
    user_stats = df_expanded.groupby('user_id').agg(
        Review_Count=('review_id', 'count'),
        Avg_Rating=('rating', 'mean')
    ).reset_index()
    
    # Segment users
    def segment_users(count):
        if count >= 10:
            return 'Power User (10+)'
        elif count >= 5:
            return 'Active User (5-9)'
        elif count >= 2:
            return 'Occasional User (2-4)'
        else:
            return 'One-time Reviewer (1)'
    
    user_stats['Segment'] = user_stats['Review_Count'].apply(segment_users)
    
    # Calculate segment statistics
    segment_reviews = df_expanded.merge(
        user_stats[['user_id', 'Segment']],
        on='user_id'
    ).groupby('Segment').size().reset_index(name='Total_Reviews')
    
    total_reviews = segment_reviews['Total_Reviews'].sum()
    segment_reviews['Percentage'] = (segment_reviews['Total_Reviews'] / total_reviews * 100).round(2)
    
    segment_order = ['Power User (10+)', 'Active User (5-9)', 
                     'Occasional User (2-4)', 'One-time Reviewer (1)']
    segment_reviews['Segment'] = pd.Categorical(
        segment_reviews['Segment'], 
        categories=segment_order, 
        ordered=True
    )
    segment_reviews = segment_reviews.sort_values('Segment')
    
    # Rating by segment
    rating_by_segment = user_stats.groupby('Segment').agg(
        Avg_Rating=('Avg_Rating', 'mean'),
        User_Count=('user_id', 'count')
    ).reset_index()
    rating_by_segment['Segment'] = pd.Categorical(
        rating_by_segment['Segment'],
        categories=segment_order,
        ordered=True
    )
    rating_by_segment = rating_by_segment.sort_values('Segment')
    
    # Visualizations
    fig = plt.figure(figsize=(18, 5))
    
    # Distribution histogram
    plt.subplot(1, 3, 1)
    counts, bins, patches = plt.hist(
        user_stats['Review_Count'],
        bins=30,
        color='#4ECDC4',
        alpha=0.7,
        edgecolor='black'
    )
    plt.xlabel("Number of Reviews")
    plt.ylabel("Number of Users")
    plt.title("Distribution of User Review Counts")
    plt.grid(axis='y', alpha=0.3)
    
    # Pie chart
    plt.subplot(1, 3, 2)
    colors = GLOBAL_COLORS[:4]
    pie_labels = [f"{seg}\n({reviews:,})" 
                  for seg, reviews in zip(segment_reviews['Segment'], 
                                         segment_reviews['Total_Reviews'])]
    plt.pie(
        segment_reviews['Percentage'],
        labels=pie_labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 9}
    )
    plt.title("Review Distribution by User Segment")
    
    # Box plot
    plt.subplot(1, 3, 3)
    segment_data = [
        user_stats[user_stats['Segment'] == seg]['Avg_Rating'].values
        for seg in segment_order
    ]
    box = plt.boxplot(
        segment_data,
        labels=[s.split('(')[0].strip() for s in segment_order],
        patch_artist=True,
        showmeans=True
    )
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel("Average Rating")
    plt.title("Rating Distribution by User Segment")
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    
    print("\n" + "="*80)
    print("USER SEGMENT ANALYSIS")
    print("="*80)
    print("\nReview Distribution:")
    print(segment_reviews.to_string(index=False))
    print("\nRating by Segment:")
    print(rating_by_segment.to_string(index=False))
    
    return user_stats



def generate_executive_summary(df):
    """Generate executive summary of the dataset"""
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    summary = {
        'Total Products': len(df),
        'Total Reviews': int(df['rating_count'].sum()),
        'Average Rating': df['rating'].mean(),
        'Average Discount': df['discount_percentage'].mean(),
        'Average Price': df['discounted_price'].mean(),
        'Total Categories': df['Main_Category'].nunique(),
        'Total Users': df['user_id'].nunique()
    }
    
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value:,}")
    
    high_rated_count = (df['rating'] >= 4.0).sum()
    high_rated_pct = (df['rating'] >= 4.0).mean() * 100
    print(f"\nHigh-rated products (≥4.0): {high_rated_count:,} ({high_rated_pct:.1f}%)")
    
    heavy_discount = (df['discount_percentage'] > 50).sum()
    print(f"Heavy discounts (>50%): {heavy_discount:,}")
    
    top_category = df['Main_Category'].mode()[0]
    print(f"Most reviewed category: {top_category}")
    
    avg_review_len = df['review_length'].mean()
    print(f"Average review length: {avg_review_len:.0f} characters")
    
    return summary



def prepare_data_for_cf(df):
    """Prepare data for collaborative filtering"""
    print("Preparing data for collaborative filtering...")
    
    # Expand user_id if multiple users per review
    cf_df = df.assign(user_id=df['user_id'].str.split(',')).explode('user_id')
    cf_df['user_id'] = cf_df['user_id'].str.strip()
    
    # Select required columns
    cf_df = cf_df[['user_id', 'product_id', 'rating']].copy()
    
    # Remove rows with missing values
    cf_df = cf_df.dropna()
    
    # Remove users with very few ratings (allow 1+)
    user_counts = cf_df['user_id'].value_counts()
    cf_df = cf_df[cf_df['user_id'].isin(user_counts[user_counts >= 1].index)]
    
    # Remove products rated by very few users
    product_counts = cf_df['product_id'].value_counts()
    cf_df = cf_df[cf_df['product_id'].isin(product_counts[product_counts >= 1].index)]
    
    print(f"Prepared dataset: {cf_df.shape[0]} ratings from {cf_df['user_id'].nunique()} users")
    print(f"Products: {cf_df['product_id'].nunique()}")
    
    return cf_df


def build_user_item_matrix(cf_df):
    """Build user-item rating matrix"""
    print("Building user-item matrix...")
    
    # Create user-item rating matrix
    user_item_matrix = cf_df.pivot_table(
        index='user_id',
        columns='product_id',
        values='rating',
        fill_value=0
    )
    
    sparsity = (user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100
    print(f"User-Item Matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {sparsity:.2f}%")
    
    return user_item_matrix


def compute_user_similarity(user_item_matrix):
    """Compute cosine similarity between users"""
    print("Computing user similarity matrix...")
    
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    return similarity_df


def find_similar_users(user_id, similarity_df, n_similar=10, min_similarity=0.0):
    """Find N most similar users to the target user"""
    if user_id not in similarity_df.index:
        raise ValueError(f"User {user_id} not found in dataset")
    
    # Get similarity scores and sort
    similarities = similarity_df[user_id].sort_values(ascending=False)
    
    # Exclude the user themselves and filter by minimum similarity
    similar_users = similarities[(similarities.index != user_id) & (similarities >= min_similarity)].head(n_similar)
    
    return similar_users

def recommend_products(user_id, user_item_matrix, similarity_df, n_recommendations=5, n_similar_users=10):
    """
    Collaborative Filtering - Recommend NEW products not yet rated by user.
    Ensures at least `n_recommendations` by filling with popular items if needed.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in dataset")
    
    # Get similar users
    similar_users = find_similar_users(user_id, similarity_df, n_similar_users, min_similarity=0.0)
    
    # Products already rated by target user
    user_rated_products = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
    
    recommendation_scores = {}

    for similar_user, similarity_score in similar_users.items():
        similar_user_ratings = user_item_matrix.loc[similar_user]
        rated_products = similar_user_ratings[similar_user_ratings > 0]
        
        for product, rating in rated_products.items():
            if product not in user_rated_products:
                if product not in recommendation_scores:
                    recommendation_scores[product] = {'score': 0, 'count': 0, 'ratings': []}
                
                recommendation_scores[product]['score'] += similarity_score * rating
                recommendation_scores[product]['count'] += 1
                recommendation_scores[product]['ratings'].append(rating)
    
    # Calculate average weighted scores
    for product in recommendation_scores:
        score = recommendation_scores[product]
        score['avg_score'] = score['score'] / score['count']
    
    # Sort recommendations by avg_score
    sorted_recs = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1]['avg_score'],
        reverse=True
    )
    
    results = [
        {
            'product_id': product,
            'predicted_rating': round(score['avg_score'], 2),
            'num_similar_users': score['count'],
            'avg_similar_rating': round(np.mean(score['ratings']), 2)
        }
        for product, score in sorted_recs
    ]
    
    results_df = pd.DataFrame(results)
    
    # Fill with top popular products if fewer than n_recommendations
    if len(results_df) < n_recommendations:
        # Get top products by overall rating
        top_products = (
            user_item_matrix.drop(user_id)
            .replace(0, np.nan)
            .mean()
            .sort_values(ascending=False)
        )
        # Exclude already rated
        top_products = top_products[~top_products.index.isin(user_rated_products)]
        
        for product_id in top_products.index:
            if len(results_df) >= n_recommendations:
                break
            results_df = pd.concat([
                results_df,
                pd.DataFrame([{
                    'product_id': product_id,
                    'predicted_rating': round(top_products[product_id], 2),
                    'num_similar_users': 0,
                    'avg_similar_rating': round(top_products[product_id], 2)
                }])
            ], ignore_index=True)
    
    return results_df.head(n_recommendations)



def get_user_profile(user_id, user_item_matrix):
    """Get user's rating profile"""
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found")
    
    user_ratings = user_item_matrix.loc[user_id]
    rated = user_ratings[user_ratings > 0]
    
    return {
        'total_rated': len(rated),
        'avg_rating': round(rated.mean(), 2),
        'max_rating': rated.max(),
        'min_rating': rated.min()
    }


def batch_recommend(user_ids, user_item_matrix, similarity_df, n_recommendations=5):
    """Generate recommendations for multiple users"""
    all_recommendations = {}
    
    for user_id in user_ids:
        try:
            print(f"\n--- Recommending for User: {user_id} ---")
            recs = recommend_products(user_id, user_item_matrix, similarity_df, n_recommendations)
            all_recommendations[user_id] = recs
            if len(recs) > 0:
                print(f"✓ Generated {len(recs)} recommendations")
            else:
                print("✗ No recommendations found")
        except ValueError as e:
            print(f"Error for user {user_id}: {e}")
            all_recommendations[user_id] = pd.DataFrame()
    
    return all_recommendations


def prepare_customer_features(df):
    """Prepare customer-level features for clustering"""
    print("\n" + "="*80)
    print("PREPARING CUSTOMER FEATURES FOR CLUSTERING")
    print("="*80)
    
    # Expand user_ids
    df_expanded = df.assign(user_id=df['user_id'].str.split(',')).explode('user_id')
    df_expanded['user_id'] = df_expanded['user_id'].str.strip()
    
    # Aggregate customer features (remove product_count as it's same as review_count)
    customer_features = df_expanded.groupby('user_id').agg({
        'review_id': 'count',              # Total reviews (also equals unique products)
        'rating': 'mean',                  # Average rating
        'discounted_price': ['mean', 'sum'],  # Price preference and total spending
        'discount_percentage': 'mean',     # Discount sensitivity
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = [
        'user_id', 'review_count', 'avg_rating',
        'avg_price', 'total_spending', 'avg_discount_pct'
    ]
    
    # Create derived features for better separation
    customer_features['repeat_buyer'] = (customer_features['review_count'] > 1).astype(int)
    customer_features['high_spender'] = (customer_features['total_spending'] > customer_features['total_spending'].quantile(0.75)).astype(int)
    customer_features['discount_seeker'] = (customer_features['avg_discount_pct'] > customer_features['avg_discount_pct'].quantile(0.75)).astype(int)
    customer_features['loyal_flag'] = ((customer_features['review_count'] >= 3) & 
                                        (customer_features['repeat_buyer'] == 1)).astype(int)
    
    # Log transform for skewed features to reduce impact of outliers
    customer_features['log_price'] = np.log1p(customer_features['avg_price'])
    customer_features['log_spending'] = np.log1p(customer_features['total_spending'])
    
    # Spending per review (value indicator)
    customer_features['spending_per_review'] = customer_features['total_spending'] / customer_features['review_count']
    
    print(f"\nTotal unique customers: {len(customer_features)}")
    print(f"\nCustomer distribution:")
    print(f"  One-time buyers: {(customer_features['review_count'] == 1).sum()} ({(customer_features['review_count'] == 1).sum() / len(customer_features) * 100:.1f}%)")
    print(f"  Repeat buyers: {(customer_features['review_count'] > 1).sum()} ({(customer_features['review_count'] > 1).sum() / len(customer_features) * 100:.1f}%)")
    print(f"  Loyal (3+ reviews): {(customer_features['review_count'] >= 3).sum()} ({(customer_features['review_count'] >= 3).sum() / len(customer_features) * 100:.1f}%)")
    
    print(f"\nFeature statistics:")
    print(customer_features[['review_count', 'avg_rating', 'avg_price', 'total_spending', 'avg_discount_pct']].describe())
    
    return customer_features

def perform_kmeans_clustering(customer_features, n_clusters=4):
    """Perform K-Means clustering with one-time buyers correctly labeled"""
    print("\n" + "="*80)
    print(f"K-MEANS CLUSTERING (k={n_clusters}, FIXED ONE-TIME BUYERS)")
    print("="*80)
    
    # Features for clustering
    feature_cols = [
        'review_count', 'avg_rating', 'log_price', 'log_spending', 
        'avg_discount_pct', 'repeat_buyer', 'discount_seeker', 
        'loyal_flag', 'spending_per_review'
    ]
    X = customer_features[feature_cols].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30, max_iter=500)
    customer_features['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    
    # --- OVERRIDE ONE-TIME BUYERS ---
    # Copy K-Means clusters
    customer_features['cluster_name'] = customer_features['cluster_kmeans'].copy()
    
    # Force all one-time buyers into their own cluster
    customer_features.loc[customer_features['review_count'] == 1, 'cluster_name'] = 'One-Time Buyers'
    
    # Map remaining clusters for repeat buyers
    remaining_clusters = customer_features.loc[customer_features['cluster_name'] != 'One-Time Buyers', 'cluster_kmeans'].unique()
    cluster_mapping = {}
    for i, cluster_id in enumerate(remaining_clusters):
        if i == 0:
            cluster_mapping[cluster_id] = 'Loyal Customers'
        elif i == 1:
            cluster_mapping[cluster_id] = 'Discount Chasers'
        else:
            cluster_mapping[cluster_id] = 'Casual Shoppers'
    
    customer_features.loc[customer_features['cluster_name'] != 'One-Time Buyers', 'cluster_name'] = \
        customer_features.loc[customer_features['cluster_name'] != 'One-Time Buyers', 'cluster_kmeans'].map(cluster_mapping)
    
    # --- Calculate clustering metrics ---
    silhouette = silhouette_score(X_scaled, customer_features['cluster_kmeans'])
    davies_bouldin = davies_bouldin_score(X_scaled, customer_features['cluster_kmeans'])
    print(f"\nClustering Metrics:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    # Cluster profile counts
    print("\nCluster Sizes:")
    print(customer_features['cluster_name'].value_counts())
    
    return customer_features, X_scaled, scaler

def interpret_clusters(cluster_profiles, customer_features):
    """Interpret cluster characteristics and assign meaningful names"""
    print("\n" + "="*80)
    print("CLUSTER INTERPRETATIONS")
    print("="*80)
    
    # Analyze cluster characteristics to assign names
    cluster_names = {}
    cluster_descriptions = {}
    
    # Sort clusters by different criteria for smart assignment
    high_engagement = cluster_profiles.sort_values('Avg_Reviews', ascending=False).index[0]
    high_discount = cluster_profiles.sort_values('Avg_Discount%', ascending=False).index[0]
    low_engagement = cluster_profiles[cluster_profiles['Repeat_Rate'] < 0.2].sort_values('Customer_Count', ascending=False).index[0] if len(cluster_profiles[cluster_profiles['Repeat_Rate'] < 0.2]) > 0 else None
    
    assigned = set()
    
    # Loyal Customers: Highest reviews and repeat rate
    for cluster_id in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster_id]
        if profile['Avg_Reviews'] >= 2.5 and profile['Repeat_Rate'] >= 0.8:
            cluster_names[cluster_id] = "Loyal Customers"
            cluster_descriptions[cluster_id] = "Frequent buyers with multiple purchases and high engagement"
            assigned.add(cluster_id)
            break
    
    # Discount Chasers: Highest discount sensitivity
    if high_discount not in assigned:
        profile = cluster_profiles.loc[high_discount]
        if profile['Avg_Discount%'] > cluster_profiles['Avg_Discount%'].median():
            cluster_names[high_discount] = "Discount Chasers"
            cluster_descriptions[high_discount] = "Price-sensitive shoppers seeking maximum discounts"
            assigned.add(high_discount)
    
    # One-Time Buyers: Largest cluster with low repeat rate
    if low_engagement is not None and low_engagement not in assigned:
        cluster_names[low_engagement] = "One-Time Buyers"
        cluster_descriptions[low_engagement] = "Single purchase customers with minimal engagement"
        assigned.add(low_engagement)
    
    # Casual Shoppers: Remaining cluster
    remaining = [cid for cid in cluster_profiles.index if cid not in assigned]
    for cluster_id in remaining:
        cluster_names[cluster_id] = "Casual Shoppers"
        cluster_descriptions[cluster_id] = "Occasional buyers with moderate spending patterns"
    
    # Fallback: ensure all clusters have names
    target_names = ["Loyal Customers", "Discount Chasers", "One-Time Buyers", "Casual Shoppers"]
    existing_names = set(cluster_names.values())
    
    if len(existing_names) < 4:
        missing = [n for n in target_names if n not in existing_names]
        unassigned = [cid for cid in cluster_profiles.index if cid not in cluster_names]
        
        for cluster_id, name in zip(unassigned, missing):
            cluster_names[cluster_id] = name
            if name == "Casual Shoppers":
                cluster_descriptions[cluster_id] = "Moderate engagement shoppers"
            elif name == "One-Time Buyers":
                cluster_descriptions[cluster_id] = "Limited purchase history"
            elif name == "Discount Chasers":
                cluster_descriptions[cluster_id] = "Price-conscious customers"
            elif name == "Loyal Customers":
                cluster_descriptions[cluster_id] = "Regular engaged buyers"
    
    # Add cluster names to customer_features
    customer_features['cluster_name'] = customer_features['cluster_kmeans'].map(cluster_names)
    
    # Print interpretations sorted by cluster size
    sorted_clusters = cluster_profiles.sort_values('Customer_Count', ascending=False).index
    
    for cluster_id in sorted_clusters:
        profile = cluster_profiles.loc[cluster_id]
        cluster_type = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        desc = cluster_descriptions.get(cluster_id, "")
        
        print(f"\n{cluster_type} (Cluster {cluster_id}):")
        print(f"  {desc}")
        print(f"  Size: {int(profile['Customer_Count'])} customers ({profile['% of Total']:.1f}%)")
        print(f"  Avg Reviews: {profile['Avg_Reviews']:.2f}")
        print(f"  Repeat Buyer Rate: {profile['Repeat_Rate']:.1%}")
        print(f"  Avg Rating: {profile['Avg_Rating']:.2f}")
        print(f"  Avg Price: ₹{profile['Avg_Price']:.2f}")
        print(f"  Total Spending: ₹{profile['Total_Spending']:.2f}")
        print(f"  Spending/Review: ₹{profile['Spending/Review']:.2f}")
        print(f"  Avg Discount: {profile['Avg_Discount%']:.1f}%")
    
    print("="*80)
    
    return cluster_names

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def visualize_clusters_enhanced(customer_features, X_scaled):
    """Enhanced cluster visualization with correct one-time buyer counts"""
    print("\n" + "="*80)
    print("VISUALIZING CLUSTERS")
    print("="*80)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    customer_features['pca1'] = X_pca[:, 0]
    customer_features['pca2'] = X_pca[:, 1]
    
    total_variance = pca.explained_variance_ratio_.sum()
    print(f"\nExplained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")
    print(f"Total variance explained by 2 PCs: {total_variance:.2%}")
    
    # Define colors and map cluster names
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    cluster_names = customer_features['cluster_name'].unique()
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(cluster_names)}
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. PCA scatter plot (LARGE)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
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
    
    ax1.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)',
                   fontweight='bold', fontsize=12)
    ax1.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)',
                   fontweight='bold', fontsize=12)
    ax1.set_title('Customer Segmentation: PCA Projection\nClusters based on behavior patterns',
                  fontweight='bold', fontsize=14, pad=15)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    
    # 2. Cluster size distribution
    ax2 = fig.add_subplot(gs[0, 2])
    cluster_sizes = customer_features['cluster_name'].value_counts()
    bars = ax2.bar(
        range(len(cluster_sizes)), cluster_sizes.values,
        color=[color_map[name] for name in cluster_sizes.index],
        edgecolor='black', linewidth=1.5, alpha=0.8
    )
    ax2.set_xlabel('Cluster', fontweight='bold')
    ax2.set_ylabel('Number of Customers', fontweight='bold')
    ax2.set_title('Cluster Size Distribution', fontweight='bold', fontsize=11)
    ax2.set_xticks(range(len(cluster_sizes)))
    ax2.set_xticklabels(cluster_sizes.index, rotation=45, ha='right')
    ax2.grid(True, alpha=0.2, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, cluster_sizes.values)):
        ax2.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + max(cluster_sizes.values)*0.02,
            f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig('customer_clusters_visualization.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    print("\n✓ Visualization saved as 'customer_clusters_visualization.png'")
    
    return pca 

def create_clduster_summary_chart(customer_features):
    """Create a detailed cluster profile heatmap"""
    # Calculate cluster profiles with cluster names (without product_count)
    if 'cluster_name' in customer_features.columns:
        cluster_profiles = customer_features.groupby('cluster_name').agg({
            'review_count': 'mean',
            'avg_rating': 'mean',
            'avg_price': 'mean',
            'total_spending': 'mean',
            'spending_per_review': 'mean',
            'avg_discount_pct': 'mean',
            'repeat_buyer': 'mean'
        })
    else:
        cluster_profiles = customer_features.groupby('cluster_kmeans').agg({
            'review_count': 'mean',
            'avg_rating': 'mean',
            'avg_price': 'mean',
            'total_spending': 'mean',
            'spending_per_review': 'mean',
            'avg_discount_pct': 'mean',
            'repeat_buyer': 'mean'
        })
    
    # Rename for display
    cluster_profiles.index.name = 'Segment'
    
    # Normalize for heatmap (0-1 scale)
    cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(cluster_profiles_norm.T, annot=cluster_profiles.T.values,
                fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Score'},
                linewidths=2, linecolor='white', ax=ax,
                yticklabels=['Avg Reviews', 'Avg Rating', 'Avg Price (₹)', 
                            'Total Spending (₹)', 'Spending/Review (₹)', 
                            'Avg Discount %', 'Repeat Rate'])
    
    ax.set_title('Cluster Profile Heatmap\n(Numbers show actual values, colors show relative intensity)',
                fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Customer Segments', fontweight='bold', fontsize=12)
    ax.set_ylabel('Behavioral Metrics', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('cluster_profile_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    print("✓ Cluster profile heatmap saved as 'cluster_profile_heatmap.png'")
    return fig



if __name__ == "__main__":
    # LOAD & EXPLORE DATA
    df, extract_path, excel_path = load_kaggle_dataset(
        "karkavelrajaj/amazon-sales-dataset",
        "amazon.csv",
        "amazon_data.xlsx"
    )
    df = explore_dataframe(df)

    # CLEAN DATA & EXPLORE DATA
    df = clean_amazon_data(df)
    df = explore_dataframe(df)

    # Generate summary
    generate_executive_summary(df)
    


    # Run analyses
    print("RUNNING ANALYSES")
    
    q1_category_distribution(df)
    q2_discount_impact(df)
    q3_user_behavior_patterns(df) 
    
    input("\nPress Enter to close all plots and exit...")
    plt.close('all')

    #COLLABORATIVE FILTERING RECOMMENDER SYSTEM

    # Prepare for collaborative filtering
    cf_df = prepare_data_for_cf(df)
    user_item_matrix = build_user_item_matrix(cf_df)
    similarity_df = compute_user_similarity(user_item_matrix)
    
    # User ID
    user_id = "AHFQGP45QKIEFKYOCYUH4DP63XGQ"

    # Get top 5 recommendations
    recommendations = recommend_products(
        user_id, user_item_matrix, similarity_df, n_recommendations=5
    )

    # Show only product IDs and predicted ratings
    top_products = recommendations[['product_id']]
    print(f"Top 5 recommended products for user {user_id}:\n")
    print(top_products.to_string(index=False))

    
    # === CUSTOMER CLUSTERING PIPELINE ===
    # Prepare customer features
    customer_features = prepare_customer_features(df)
    
    # Standardize features (without product_count, added new features)
    X_scaled = StandardScaler().fit_transform(
        customer_features[[
            'review_count', 'avg_rating', 'log_price', 'log_spending', 
            'avg_discount_pct', 'repeat_buyer', 'discount_seeker', 
            'loyal_flag', 'spending_per_review'
        ]]
    )
    
    # Perform clustering with k=4
    customer_features, X_scaled, scaler = perform_kmeans_clustering(
        customer_features, n_clusters=4
    )
    
    # Visualize clusters
    pca = visualize_clusters_enhanced(customer_features, X_scaled)
    
    # Save results
    create_cluster_summary_chart(customer_features)
    
    # Save final customer features with clusters
    output_path = os.path.join(os.getcwd(), "customer_clusters.xlsx")
    customer_features.to_excel(output_path, index=False)
    print(f"\n✓ Customer cluster data saved: {output_path}")
    
    print("✓ CLUSTERING ANALYSIS COMPLETE")    
    input("\nPress Enter to close all plots and exit...")

    plt.close('all')


