import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pickle

# Load processed data
df = pd.read_csv('processed_products.csv')

## Content-Based Filtering
class ContentBasedRecommender:
    def __init__(self, data):
        self.df = data
        self.similarity_matrix = None
        
    def prepare_similarity_matrix(self):
        features = ['Price', 'Rating', 'Main Category', 'Sub Category', 
                   'Discount', 'Popularity']
        feature_matrix = self.df[features]
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
    def recommend_products(self, product_name, n=5):
        if self.similarity_matrix is None:
            self.prepare_similarity_matrix()
            
        idx = self.df[self.df['Product Name'] == product_name].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        product_indices = [i[0] for i in sim_scores]
        return self.df.iloc[product_indices]

## Collaborative Filtering (User-Item Matrix approach)
class CollaborativeFilteringRecommender:
    def __init__(self, data):
        self.df = data
        self.model = None
        self.user_item_matrix = None
        
    def create_user_item_matrix(self):
        # Create synthetic user-item interactions
        user_ids = []
        product_ids = []
        ratings = []
        
        for _, row in self.df.iterrows():
            num_users = max(1, int(row['Number of Buyers'] / 1000))  # Scale down
            for user_num in range(num_users):
                user_id = f"user_{user_num}"
                rating = max(1, min(5, row['Rating'] * 5 + np.random.normal(0, 0.5)))
                user_ids.append(user_id)
                product_ids.append(row['Product Name'])
                ratings.append(rating)
        
        # Create DataFrame
        interactions = pd.DataFrame({
            'user_id': user_ids,
            'product_id': product_ids,
            'rating': ratings
        })
        
        # Create pivot table
        self.user_item_matrix = interactions.pivot_table(
            index='user_id', 
            columns='product_id', 
            values='rating'
        ).fillna(0)
        
        # Convert to sparse matrix
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
    def train_model(self):
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
            
        # KNN model for user-based collaborative filtering
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        
    def recommend_products(self, user_id, n=5):
        if self.model is None:
            self.train_model()
            
        # Find similar users
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        distances, indices = self.model.kneighbors(
            self.sparse_matrix[user_idx], 
            n_neighbors=n+1
        )
        
        # Get products from similar users
        similar_users = self.user_item_matrix.iloc[indices[0]].index[1:]
        recommendations = self.user_item_matrix.loc[similar_users].mean(axis=0)
        recommendations = recommendations.sort_values(ascending=False)[:n]
        
        # Get product details
        recommended_products = []
        for product_id in recommendations.index:
            product_details = self.df[self.df['Product Name'] == product_id].iloc[0]
            recommended_products.append(product_details)
            
        return pd.DataFrame(recommended_products)

# Initialize recommenders
content_rec = ContentBasedRecommender(df)
content_rec.prepare_similarity_matrix()

collab_rec = CollaborativeFilteringRecommender(df)
collab_rec.train_model()

# Save models
with open('content_based_model.pkl', 'wb') as f:
    pickle.dump(content_rec, f)

with open('collab_filter_model.pkl', 'wb') as f:
    pickle.dump(collab_rec, f)