"""
Core Matrix Mining Algorithms
Implements SVD, NMF, and PCA for item recommendation system
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import pandas as pd

class MatrixMiner:
    def __init__(self, ratings_matrix, item_df, user_df):
        """
        Initialize matrix mining system
        
        Args:
            ratings_matrix: User-item rating matrix (students × courses)
            item_df: Item metadata
            user_df: User metadata
        """
        self.ratings_matrix = ratings_matrix
        self.item_df = item_df
        self.user_df = user_df
        self.n_users, self.n_items = ratings_matrix.shape
        
        # Models
        self.svd_model = None
        self.nmf_model = None
        self.pca_model = None
        
        # Transformed data
        self.svd_student_factors = None
        self.svd_course_factors = None
        self.nmf_student_factors = None
        self.nmf_course_factors = None
        self.pca_transformed = None
        
    def apply_svd(self, n_components=10, log_callback=None):
        """
        Apply Singular Value Decomposition for collaborative filtering
        
        Args:
            n_components: Number of latent factors
            log_callback: Optional callable(msg: str) for real-time progress
            
        Returns:
            dict with SVD results and metrics
        """
        def log(msg, level='info'):
            print(msg)
            if log_callback:
                log_callback(msg, level)

        log(f"[>>] Starting SVD with {n_components} components...", 'info')
        log(f"  Matrix shape: {self.ratings_matrix.shape[0]} users x {self.ratings_matrix.shape[1]} items", 'info')
        
        # Create sparse matrix for efficiency
        log("  Converting to sparse matrix...", 'info')
        sparse_ratings = csr_matrix(self.ratings_matrix)
        
        # Apply SVD
        log(f"  Fitting TruncatedSVD model (k={n_components})...", 'info')
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_student_factors = self.svd_model.fit_transform(sparse_ratings)
        self.svd_course_factors = self.svd_model.components_.T
        
        # Calculate explained variance
        log("  Computing explained variance ratios...", 'info')
        explained_variance_ratio = self.svd_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Reconstruct matrix for evaluation
        log("  Reconstructing rating matrix...", 'info')
        reconstructed = self.svd_student_factors @ self.svd_model.components_
        
        # Calculate RMSE on non-zero entries
        mask = self.ratings_matrix > 0
        rmse = np.sqrt(np.mean((self.ratings_matrix[mask] - reconstructed[mask])**2))
        
        results = {
            'n_components': n_components,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'total_variance_explained': cumulative_variance[-1],
            'rmse': rmse,
            'singular_values': self.svd_model.singular_values_,
            'user_factors': self.svd_student_factors,
            'item_factors': self.svd_course_factors
        }
        
        log(f"[OK] SVD complete - Variance explained: {cumulative_variance[-1]*100:.2f}%", 'success')
        log(f"  Reconstruction RMSE: {rmse:.4f}", 'success')
        log(f"  Top singular values: {', '.join(f'{sv:.2f}' for sv in self.svd_model.singular_values_[:5])}", 'success')
        
        return results
    
    def apply_nmf(self, n_components=10, max_iter=200, log_callback=None):
        """
        Apply Non-negative Matrix Factorization
        
        Args:
            n_components: Number of latent factors
            max_iter: Maximum iterations
            log_callback: Optional callable(msg: str, level: str) for real-time progress
            
        Returns:
            dict with NMF results and metrics
        """
        def log(msg, level='info'):
            print(msg)
            if log_callback:
                log_callback(msg, level)

        log(f"[>>] Starting NMF with {n_components} components (max_iter={max_iter})...", 'info')
        log(f"  Matrix shape: {self.ratings_matrix.shape[0]} users x {self.ratings_matrix.shape[1]} items", 'info')
        
        # NMF requires non-negative values
        log("  Preparing non-negative rating matrix...", 'info')
        ratings_nonneg = self.ratings_matrix.copy()
        
        # Apply NMF
        log(f"  Fitting NMF model (init=random, random_state=42)...", 'info')
        self.nmf_model = NMF(n_components=n_components, init='random', 
                             random_state=42, max_iter=max_iter)
        self.nmf_student_factors = self.nmf_model.fit_transform(ratings_nonneg)
        self.nmf_course_factors = self.nmf_model.components_.T
        
        log(f"  Factorization converged in {self.nmf_model.n_iter_} iterations", 'info')
        log(f"  User factors shape: {self.nmf_student_factors.shape}", 'info')
        log(f"  Item factors shape: {self.nmf_course_factors.shape}", 'info')
        
        # Reconstruct matrix
        log("  Reconstructing rating matrix...", 'info')
        reconstructed = self.nmf_student_factors @ self.nmf_model.components_
        
        # Calculate RMSE on non-zero entries
        mask = self.ratings_matrix > 0
        rmse = np.sqrt(np.mean((self.ratings_matrix[mask] - reconstructed[mask])**2))
        
        # Calculate reconstruction error
        reconstruction_error = self.nmf_model.reconstruction_err_
        
        results = {
            'n_components': n_components,
            'reconstruction_error': reconstruction_error,
            'rmse': rmse,
            'n_iter': self.nmf_model.n_iter_,
            'user_factors': self.nmf_student_factors,
            'item_factors': self.nmf_course_factors
        }
        
        log(f"[OK] NMF complete - {self.nmf_model.n_iter_} iterations", 'success')
        log(f"  Reconstruction RMSE: {rmse:.4f}", 'success')
        log(f"  Reconstruction error: {reconstruction_error:.4f}", 'success')
        
        return results
    
    def apply_pca(self, n_components=10, log_callback=None):
        """
        Apply PCA for dimensionality reduction and visualization
        
        Args:
            n_components: Number of principal components
            log_callback: Optional callable(msg: str, level: str) for real-time progress
            
        Returns:
            dict with PCA results
        """
        def log(msg, level='info'):
            print(msg)
            if log_callback:
                log_callback(msg, level)

        log(f"[>>] Starting PCA with {n_components} components...", 'info')
        log(f"  Matrix shape: {self.ratings_matrix.shape[0]} users x {self.ratings_matrix.shape[1]} items", 'info')
        
        # Standardize the data
        log("  Standardizing rating matrix (StandardScaler)...", 'info')
        scaler = StandardScaler()
        ratings_scaled = scaler.fit_transform(self.ratings_matrix)
        
        # Apply PCA
        log(f"  Fitting PCA model (k={n_components})...", 'info')
        self.pca_model = PCA(n_components=n_components, random_state=42)
        self.pca_transformed = self.pca_model.fit_transform(ratings_scaled)
        
        # Get explained variance
        log("  Computing principal components...", 'info')
        explained_variance_ratio = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        results = {
            'n_components': n_components,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'total_variance_explained': cumulative_variance[-1],
            'principal_components': self.pca_model.components_,
            'transformed_data': self.pca_transformed
        }
        
        log(f"[OK] PCA complete - Variance explained: {cumulative_variance[-1]*100:.2f}%", 'success')
        for i, (var, cum) in enumerate(zip(explained_variance_ratio[:5], cumulative_variance[:5])):
            log(f"  PC{i+1}: {var*100:.2f}% individual, {cum*100:.2f}% cumulative", 'success')
        
        return results
    
    def recommend_items_svd(self, user_idx, n_recommendations=10, exclude_rated=True):
        """
        Recommend courses using SVD-based collaborative filtering
        
        Args:
            user_idx: User index
            n_recommendations: Number of courses to recommend
            exclude_rated: Whether to exclude already rated courses
            
        Returns:
            DataFrame with recommended courses
        """
        if self.svd_model is None:
            raise ValueError("SVD model not trained. Call apply_svd() first.")
        
        # Predict ratings for all courses
        predicted_ratings = self.svd_student_factors[user_idx] @ self.svd_model.components_
        
        # Exclude already rated courses if requested
        if exclude_rated:
            rated_mask = self.ratings_matrix[user_idx] > 0
            predicted_ratings[rated_mask] = -np.inf
        
        # Get top N recommendations
        top_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        
        # Create recommendations dataframe
        recommendations = []
        for idx in top_indices:
            item_info = self.item_df.iloc[idx].to_dict()
            item_info['predicted_rating'] = predicted_ratings[idx]
            item_info['actual_rating'] = self.ratings_matrix[user_idx, idx]
            recommendations.append(item_info)
        
        return pd.DataFrame(recommendations)
    
    def recommend_items_nmf(self, user_idx, n_recommendations=10, exclude_rated=True):
        """
        Recommend courses using NMF-based collaborative filtering
        
        Args:
            user_idx: User index
            n_recommendations: Number of courses to recommend
            exclude_rated: Whether to exclude already rated courses
            
        Returns:
            DataFrame with recommended courses
        """
        if self.nmf_model is None:
            raise ValueError("NMF model not trained. Call apply_nmf() first.")
        
        # Predict ratings for all courses
        predicted_ratings = self.nmf_student_factors[user_idx] @ self.nmf_model.components_
        
        # Exclude already rated courses if requested
        if exclude_rated:
            rated_mask = self.ratings_matrix[user_idx] > 0
            predicted_ratings[rated_mask] = -np.inf
        
        # Get top N recommendations
        top_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        
        # Create recommendations dataframe
        recommendations = []
        for idx in top_indices:
            item_info = self.item_df.iloc[idx].to_dict()
            item_info['predicted_rating'] = predicted_ratings[idx]
            item_info['actual_rating'] = self.ratings_matrix[user_idx, idx]
            recommendations.append(item_info)
        
        return pd.DataFrame(recommendations)
    
    def find_similar_items(self, item_idx, n_similar=10, method='svd'):
        """
        Find similar courses based on latent factors
        
        Args:
            item_idx: Item index
            n_similar: Number of similar courses to return
            method: 'svd' or 'nmf'
            
        Returns:
            DataFrame with similar courses
        """
        if method == 'svd':
            if self.svd_course_factors is None:
                raise ValueError("SVD not applied yet")
            item_factors = self.svd_course_factors
        else:
            if self.nmf_course_factors is None:
                raise ValueError("NMF not applied yet")
            item_factors = self.nmf_course_factors
        
        # Calculate cosine similarity
        target_vector = item_factors[item_idx]
        similarities = item_factors @ target_vector
        similarities = similarities / (np.linalg.norm(item_factors, axis=1) * np.linalg.norm(target_vector))
        
        # Get top N similar courses (excluding itself)
        similarities[item_idx] = -np.inf
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        
        # Create results dataframe
        similar_items = []
        for idx in top_indices:
            item_info = self.item_df.iloc[idx].to_dict()
            item_info['similarity'] = similarities[idx]
            similar_items.append(item_info)
        
        return pd.DataFrame(similar_items)
    
    def get_user_profile(self, user_idx, method='svd'):
        """
        Get user's latent factor profile
        
        Args:
            user_idx: User index
            method: 'svd' or 'nmf'
            
        Returns:
            User's latent factors
        """
        if method == 'svd':
            return self.svd_student_factors[user_idx] if self.svd_student_factors is not None else None
        else:
            return self.nmf_student_factors[user_idx] if self.nmf_student_factors is not None else None
