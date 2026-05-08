"""
Dataset Generator for Recommender System
Generates synthetic user-item rating data
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

class AcademicDataGenerator:
    def __init__(self, n_users=15, n_items=10000, sparsity=0.95):
        """
        Initialize data generator
        
        Args:
            n_users: Number of users (minimum 15)
            n_items: Number of items (minimum 10000)
            sparsity: Proportion of missing ratings (realistic for recommendation data)
        """
        self.n_users = max(n_users, 15)
        self.n_items = max(n_items, 10000)
        self.sparsity = sparsity
        
    def generate_course_features(self):
        """Generate item metadata"""
        categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        tiers = [1, 2, 3, 4, 5]
        
        courses = []
        for i in range(self.n_items):
            category = np.random.choice(categories)
            tier = np.random.choice(tiers)
            item_id = f"{category}{tier}_{i:05d}"
            popularity = np.random.uniform(1, 5)
            quality = np.random.uniform(1, 5)
            
            courses.append({
                'item_id': item_id,
                'category': category,
                'tier': tier,
                'popularity': popularity,
                'quality': quality
            })
        
        return pd.DataFrame(courses)
    
    def generate_student_profiles(self):
        """Generate user metadata"""
        segments = ['Segment A', 'Segment B', 'Segment C', 'Segment D', 
                    'Segment E', 'Segment F', 'Segment G']
        
        students = []
        for i in range(self.n_users):
            user_id = f"USER_{i+1:03d}"
            segment = np.random.choice(segments)
            activity_score = np.random.uniform(0.0, 1.0)
            engagement_level = np.random.choice([1, 2, 3, 4, 5])
            
            students.append({
                'user_id': user_id,
                'segment': segment,
                'activity_score': activity_score,
                'engagement_level': engagement_level
            })
        
        return pd.DataFrame(students)
    
    def generate_ratings_matrix(self, item_df, user_df):
        """
        Generate user-item rating matrix with realistic patterns
        """
        # Initialize empty matrix
        ratings = np.zeros((self.n_users, self.n_items))
        
        # Create latent factors for realistic patterns
        n_factors = 5
        user_factors = np.random.randn(self.n_users, n_factors)
        item_factors = np.random.randn(n_factors, self.n_items)
        
        # Generate base ratings from latent factors
        base_ratings = user_factors @ item_factors
        base_ratings = (base_ratings - base_ratings.min()) / (base_ratings.max() - base_ratings.min())
        base_ratings = base_ratings * 4 + 1  # Scale to 1-5
        
        # Add noise
        noise = np.random.normal(0, 0.5, (self.n_users, self.n_items))
        ratings = np.clip(base_ratings + noise, 1, 5)
        
        # Apply sparsity (users only rate items they've interacted with)
        mask = np.random.random((self.n_users, self.n_items)) > self.sparsity
        ratings = np.where(mask, ratings, 0)
        
        # Ensure each user has rated at least some items
        for i in range(self.n_users):
            if ratings[i].sum() == 0:
                n_rated = np.random.randint(5, 20)
                rated_indices = np.random.choice(self.n_items, n_rated, replace=False)
                ratings[i, rated_indices] = np.random.uniform(1, 5, n_rated)
        
        return ratings
    
    def generate_dataset(self, output_dir='data'):
        """Generate complete dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating dataset with {self.n_users} users and {self.n_items} items...")
        
        # Generate components
        item_df = self.generate_course_features()
        user_df = self.generate_student_profiles()
        ratings_matrix = self.generate_ratings_matrix(item_df, user_df)
        
        # Save data
        item_df.to_csv(f'{output_dir}/courses.csv', index=False)
        user_df.to_csv(f'{output_dir}/students.csv', index=False)
        np.save(f'{output_dir}/ratings_matrix.npy', ratings_matrix)
        
        # Create ratings dataframe (sparse format)
        ratings_list = []
        for i in range(self.n_users):
            for j in range(self.n_items):
                if ratings_matrix[i, j] > 0:
                    ratings_list.append({
                        'user_id': user_df.iloc[i]['user_id'],
                        'item_id': item_df.iloc[j]['item_id'],
                        'rating': ratings_matrix[i, j]
                    })
        
        ratings_df = pd.DataFrame(ratings_list)
        ratings_df.to_csv(f'{output_dir}/ratings.csv', index=False)
        
        # Generate data dictionary
        self._generate_data_dictionary(output_dir, item_df, user_df, ratings_df)
        
        print(f"Dataset generated successfully!")
        print(f"  - Users: {self.n_users}")
        print(f"  - Items: {self.n_items}")
        print(f"  - Total ratings: {len(ratings_list)}")
        print(f"  - Sparsity: {(1 - len(ratings_list)/(self.n_users * self.n_items))*100:.2f}%")
        
        return item_df, user_df, ratings_matrix
    
    def _generate_data_dictionary(self, output_dir, item_df, user_df, ratings_df):
        """Generate data dictionary documentation"""
        dictionary = f"""# Data Dictionary
## MineMatrix Recommender System Dataset

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source:** Synthetic data for recommendation system testing

### Dataset Specifications
- **Rows (Users):** {self.n_users}
- **Columns (Items):** {self.n_items}
- **Total Ratings:** {len(ratings_df)}
- **Sparsity:** {(1 - len(ratings_df)/(self.n_users * self.n_items))*100:.2f}%

### Files

#### 1. students.csv
User profile information
- `user_id`: Unique user identifier (USER_001 to USER_{self.n_users:03d})
- `segment`: User segment classification
- `activity_score`: User activity level (0.0-1.0)
- `engagement_level`: User engagement level (1-5)

#### 2. courses.csv
Item catalog information
- `item_id`: Unique item identifier (CAT#_#####)
- `category`: Item category code
- `tier`: Item tier level (1-5)
- `popularity`: Item popularity score (1-5)
- `quality`: Item quality score (1-5)

#### 3. ratings.csv
User item ratings (sparse format)
- `user_id`: User identifier
- `item_id`: Item identifier
- `rating`: Rating score (1-5 scale)

#### 4. ratings_matrix.npy
Dense matrix format (users × items)
- Rows: Users
- Columns: Items
- Values: Ratings (0 = not rated)

### Data Collection Methodology
This synthetic dataset simulates realistic user-item rating patterns using:
1. Latent factor models to create user-item affinity patterns
2. Category and tier-based item clustering
3. Realistic sparsity (users only rate items they've interacted with)
4. Gaussian noise to simulate rating variability

### Usage Notes
- Zero values in the matrix indicate unrated items (not low ratings)
- Ratings follow a 1-5 scale (1=Poor, 5=Excellent)
- Dataset designed for collaborative filtering and matrix factorization experiments
"""
        
        with open(f'{output_dir}/DATA_DICTIONARY.md', 'w') as f:
            f.write(dictionary)

if __name__ == '__main__':
    generator = AcademicDataGenerator(n_users=15, n_items=10000, sparsity=0.95)
    generator.generate_dataset()
