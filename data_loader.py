"""
Data Loader for External Datasets
Supports loading datasets from Kaggle, CSV files, or other sources
"""

import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix

class ExternalDataLoader:
    """
    Load and preprocess external datasets for matrix mining
    Supports various formats: CSV, ratings format, user-item matrices
    """
    
    def __init__(self):
        self.item_df = None
        self.user_df = None
        self.ratings_matrix = None
        self.ratings_df = None
        
    def load_from_ratings_file(self, filepath, user_col='user_id', item_col='item_id', 
                                rating_col='rating', separator=','):
        """
        Load dataset from ratings file (user, item, rating format)
        Common for MovieLens, Amazon, etc.
        
        Args:
            filepath: Path to ratings file
            user_col: Name of user/user column
            item_col: Name of item/item column
            rating_col: Name of rating column
            separator: CSV separator (default: ',')
            
        Returns:
            tuple: (item_df, user_df, ratings_matrix)
        """
        print(f"Loading ratings from: {filepath}")
        
        # Load ratings
        self.ratings_df = pd.read_csv(filepath, sep=separator, encoding='utf-8-sig')
        
        # Validate columns
        if user_col not in self.ratings_df.columns:
            raise ValueError(f"Column '{user_col}' not found. Available: {self.ratings_df.columns.tolist()}")
        if item_col not in self.ratings_df.columns:
            raise ValueError(f"Column '{item_col}' not found. Available: {self.ratings_df.columns.tolist()}")
        if rating_col not in self.ratings_df.columns:
            raise ValueError(f"Column '{rating_col}' not found. Available: {self.ratings_df.columns.tolist()}")
        
        # Rename columns for consistency
        self.ratings_df = self.ratings_df.rename(columns={
            user_col: 'user_id',
            item_col: 'item_id',
            rating_col: 'rating'
        })
        
        # Create user dataframe (only real columns — no fake defaults)
        unique_users = self.ratings_df['user_id'].unique()
        self.user_df = pd.DataFrame({'user_id': unique_users})
        
        # Create item dataframe (only real columns — no fake defaults)
        unique_items = self.ratings_df['item_id'].unique()
        self.item_df = pd.DataFrame({'item_id': unique_items})
        
        # Create ratings matrix
        self.ratings_matrix = self._create_matrix()
        
        print(f"✓ Loaded: {len(self.user_df)} users × {len(self.item_df)} items")
        print(f"  Total ratings: {len(self.ratings_df)}")
        print(f"  Sparsity: {(1 - len(self.ratings_df)/(len(self.user_df)*len(self.item_df)))*100:.2f}%")
        
        return self.item_df, self.user_df, self.ratings_matrix
    
    def load_from_matrix_file(self, filepath, has_header=True, has_index=True):
        """
        Load dataset from matrix file (users as rows, items as columns)
        
        Args:
            filepath: Path to matrix file (CSV)
            has_header: Whether file has column headers
            has_index: Whether file has row index
            
        Returns:
            tuple: (item_df, user_df, ratings_matrix)
        """
        print(f"Loading matrix from: {filepath}")
        
        # Load matrix
        if has_index:
            df = pd.read_csv(filepath, index_col=0, encoding='utf-8-sig')
        else:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        self.ratings_matrix = df.values
        
        # Create user dataframe
        if has_index:
            student_ids = df.index.tolist()
        else:
            student_ids = [f"USER_{i+1}" for i in range(len(df))]
        
        self.user_df = pd.DataFrame({
            'user_id': student_ids,
            'major': 'Unknown',
            'gpa': 3.5,
            'year': 3
        })
        
        # Create item dataframe
        if has_header:
            course_ids = df.columns.tolist()
        else:
            course_ids = [f"ITEM_{i+1}" for i in range(df.shape[1])]
        
        self.item_df = pd.DataFrame({
            'item_id': course_ids,
            'department': 'General',
            'level': 300,
            'difficulty': 3.0,
            'workload': 3.0
        })
        
        print(f"✓ Loaded: {len(self.user_df)} users × {len(self.item_df)} items")
        print(f"  Total ratings: {np.count_nonzero(self.ratings_matrix)}")
        print(f"  Sparsity: {(1 - np.count_nonzero(self.ratings_matrix)/self.ratings_matrix.size)*100:.2f}%")
        
        return self.item_df, self.user_df, self.ratings_matrix
    
    def load_movielens(self, ratings_file, movies_file=None, users_file=None):
        """
        Load MovieLens dataset (common format)
        
        Args:
            ratings_file: Path to ratings.csv
            movies_file: Path to movies.csv (optional)
            users_file: Path to users.csv (optional)
            
        Returns:
            tuple: (item_df, user_df, ratings_matrix)
        """
        print("Loading MovieLens dataset...")
        
        # Load ratings
        self.ratings_df = pd.read_csv(ratings_file, encoding='utf-8-sig')
        
        # MovieLens typically has: userId, movieId, rating, timestamp
        if 'userId' in self.ratings_df.columns:
            self.ratings_df = self.ratings_df.rename(columns={
                'userId': 'user_id',
                'movieId': 'item_id'
            })
        
        # Load movies if available
        if movies_file and os.path.exists(movies_file):
            movies_df = pd.read_csv(movies_file, encoding='utf-8-sig')
            if 'movieId' in movies_df.columns:
                movies_df = movies_df.rename(columns={'movieId': 'item_id'})
            
            # Extract genre as department
            if 'genres' in movies_df.columns:
                movies_df['department'] = movies_df['genres'].str.split('|').str[0]
            
            self.item_df = movies_df[['item_id']].copy()
            if 'title' in movies_df.columns:
                self.item_df['title'] = movies_df['title']
            if 'department' in movies_df.columns:
                self.item_df['department'] = movies_df['department']
            # Keep any other real columns from movies file (genres, year, etc.)
            for col in movies_df.columns:
                if col not in ['item_id'] and col not in self.item_df.columns:
                    self.item_df[col] = movies_df[col].values
        else:
            # No movies file — just item IDs
            unique_items = self.ratings_df['item_id'].unique()
            self.item_df = pd.DataFrame({'item_id': unique_items})
        
        # Load users if available (no fake defaults — only real columns)
        if users_file and os.path.exists(users_file):
            users_df = pd.read_csv(users_file, encoding='utf-8-sig')
            if 'userId' in users_df.columns:
                users_df = users_df.rename(columns={'userId': 'user_id'})
            self.user_df = users_df
        else:
            # Only user_id — let the dataset speak for itself
            unique_users = self.ratings_df['user_id'].unique()
            self.user_df = pd.DataFrame({'user_id': unique_users})
        
        # Create ratings matrix
        self.ratings_matrix = self._create_matrix()
        
        print(f"✓ Loaded MovieLens: {len(self.user_df)} users × {len(self.item_df)} movies")
        print(f"  Total ratings: {len(self.ratings_df)}")
        
        return self.item_df, self.user_df, self.ratings_matrix
    
    def _create_matrix(self):
        """Create ratings matrix from ratings dataframe"""
        # Create mappings
        user_to_idx = {sid: idx for idx, sid in enumerate(self.user_df['user_id'])}
        item_to_idx = {cid: idx for idx, cid in enumerate(self.item_df['item_id'])}
        
        # Initialize matrix
        n_users = len(self.user_df)
        n_items = len(self.item_df)
        ratings_matrix = np.zeros((n_users, n_items))
        
        # Fill matrix
        for _, row in self.ratings_df.iterrows():
            user_idx = user_to_idx.get(row['user_id'])
            item_idx = item_to_idx.get(row['item_id'])
            
            if user_idx is not None and item_idx is not None:
                ratings_matrix[user_idx, item_idx] = row['rating']
        
        return ratings_matrix
    
    def save_processed_data(self, output_dir='data'):
        """Save processed data in standard format"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.item_df.to_csv(f'{output_dir}/courses.csv', index=False)
        self.user_df.to_csv(f'{output_dir}/students.csv', index=False)
        self.ratings_df.to_csv(f'{output_dir}/ratings.csv', index=False)
        np.save(f'{output_dir}/ratings_matrix.npy', self.ratings_matrix)
        
        print(f"✓ Data saved to {output_dir}/")
    
    def add_metadata(self, student_metadata=None, course_metadata=None):
        """
        Add additional metadata to students or courses
        
        Args:
            student_metadata: DataFrame with user_id and additional columns
            course_metadata: DataFrame with item_id and additional columns
        """
        if student_metadata is not None:
            self.user_df = self.user_df.merge(student_metadata, on='user_id', how='left')
            print(f"✓ Added user metadata: {student_metadata.columns.tolist()}")
        
        if course_metadata is not None:
            self.item_df = self.item_df.merge(course_metadata, on='item_id', how='left')
            print(f"✓ Added item metadata: {course_metadata.columns.tolist()}")
    
    def filter_data(self, min_ratings_per_user=5, min_ratings_per_item=5):
        """
        Filter out users/items with too few ratings
        
        Args:
            min_ratings_per_user: Minimum ratings per user
            min_ratings_per_item: Minimum ratings per item
        """
        print(f"Filtering data (min {min_ratings_per_user} ratings/user, {min_ratings_per_item} ratings/item)...")
        
        initial_users = len(self.user_df)
        initial_items = len(self.item_df)
        
        # Filter users
        user_counts = self.ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings_per_user].index
        self.ratings_df = self.ratings_df[self.ratings_df['user_id'].isin(valid_users)]
        
        # Filter items
        item_counts = self.ratings_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_ratings_per_item].index
        self.ratings_df = self.ratings_df[self.ratings_df['item_id'].isin(valid_items)]
        
        # Update dataframes
        self.user_df = self.user_df[self.user_df['user_id'].isin(valid_users)]
        self.item_df = self.item_df[self.item_df['item_id'].isin(valid_items)]
        
        # Recreate matrix
        self.ratings_matrix = self._create_matrix()
        
        print(f"✓ Filtered: {initial_users} → {len(self.user_df)} users")
        print(f"  {initial_items} → {len(self.item_df)} items")
        print(f"  {len(self.ratings_df)} ratings remaining")
    
    def normalize_ratings(self, scale=(1, 5)):
        """
        Normalize ratings to specified scale
        
        Args:
            scale: Tuple of (min, max) for rating scale
        """
        print(f"Normalizing ratings to {scale[0]}-{scale[1]} scale...")
        
        current_min = self.ratings_df['rating'].min()
        current_max = self.ratings_df['rating'].max()
        
        # Normalize
        self.ratings_df['rating'] = (
            (self.ratings_df['rating'] - current_min) / (current_max - current_min) * 
            (scale[1] - scale[0]) + scale[0]
        )
        
        # Recreate matrix
        self.ratings_matrix = self._create_matrix()
        
        print(f"✓ Ratings normalized from [{current_min}, {current_max}] to {scale}")

def load_kaggle_dataset(dataset_path, dataset_type='ratings', **kwargs):
    """
    Convenience function to load Kaggle datasets
    
    Args:
        dataset_path: Path to dataset file or directory
        dataset_type: Type of dataset ('ratings', 'matrix', 'movielens')
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        tuple: (item_df, user_df, ratings_matrix)
    """
    loader = ExternalDataLoader()
    
    if dataset_type == 'ratings':
        return loader.load_from_ratings_file(dataset_path, **kwargs)
    elif dataset_type == 'matrix':
        return loader.load_from_matrix_file(dataset_path, **kwargs)
    elif dataset_type == 'movielens':
        return loader.load_movielens(dataset_path, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

# Example usage
if __name__ == '__main__':
    print("="*80)
    print("EXTERNAL DATA LOADER - EXAMPLES")
    print("="*80)
    
    print("\nExample 1: Load from ratings file (user, item, rating format)")
    print("-" * 80)
    print("loader = ExternalDataLoader()")
    print("item_df, user_df, matrix = loader.load_from_ratings_file(")
    print("    'path/to/ratings.csv',")
    print("    user_col='userId',")
    print("    item_col='movieId',")
    print("    rating_col='rating'")
    print(")")
    
    print("\nExample 2: Load MovieLens dataset")
    print("-" * 80)
    print("loader = ExternalDataLoader()")
    print("item_df, user_df, matrix = loader.load_movielens(")
    print("    ratings_file='ratings.csv',")
    print("    movies_file='movies.csv'")
    print(")")
    
    print("\nExample 3: Load from matrix file")
    print("-" * 80)
    print("loader = ExternalDataLoader()")
    print("item_df, user_df, matrix = loader.load_from_matrix_file(")
    print("    'path/to/matrix.csv',")
    print("    has_header=True,")
    print("    has_index=True")
    print(")")
    
    print("\nExample 4: Filter and normalize")
    print("-" * 80)
    print("loader.filter_data(min_ratings_per_user=10, min_ratings_per_item=5)")
    print("loader.normalize_ratings(scale=(1, 5))")
    print("loader.save_processed_data('data/')")
    
    print("\n" + "="*80)
    print("See KAGGLE_DATASETS.md for detailed instructions")
    print("="*80)
