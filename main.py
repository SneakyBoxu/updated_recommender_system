"""
Main Application - Matrix Mining in Academic Ecosystems
GUI application for item recommendation using SVD and NMF
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os

from data_generator import AcademicDataGenerator
from matrix_mining import MatrixMiner

class MatrixMiningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Mining in Academic Ecosystems - ISU Item Recommender")
        self.root.geometry("1200x800")
        
        # Data
        self.item_df = None
        self.user_df = None
        self.ratings_matrix = None
        self.miner = None
        self.svd_results = None
        self.nmf_results = None
        self.pca_results = None
        
        # Setup UI
        self.setup_ui()
        
        # Load or generate data
        self.load_data()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load MovieLens (ratings + movies)...", command=self.load_movielens_dialog)
        file_menu.add_command(label="Load Single Ratings CSV...", command=self.load_kaggle_dataset)
        file_menu.add_command(label="Load Custom CSV...", command=self.load_custom_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Regenerate Synthetic Data", command=self.regenerate_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Data Overview
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text='Data Overview')
        self.setup_data_tab()
        
        # Tab 2: SVD Analysis
        self.tab_svd = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_svd, text='SVD Analysis')
        self.setup_svd_tab()
        
        # Tab 3: NMF Analysis
        self.tab_nmf = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_nmf, text='NMF Analysis')
        self.setup_nmf_tab()
        
        # Tab 4: PCA Visualization
        self.tab_pca = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pca, text='PCA Visualization')
        self.setup_pca_tab()
        
        # Tab 5: Recommendations
        self.tab_recommend = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_recommend, text='Item Recommendations')
        self.setup_recommend_tab()
        
        # Bottom toolbar with Close button
        bottom_bar = ttk.Frame(self.root)
        bottom_bar.pack(fill='x', side='bottom', padx=5, pady=5)
        
        tk.Button(
            bottom_bar,
            text='✕  Close Application',
            command=self.close_app,
            bg='#c0392b',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=12,
            pady=4,
            cursor='hand2'
        ).pack(side='right')
        
    def setup_data_tab(self):
        """Setup data overview tab"""
        frame = ttk.Frame(self.tab_data, padding="10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Dataset Overview", font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.data_text = scrolledtext.ScrolledText(frame, height=30, width=100)
        self.data_text.pack(fill='both', expand=True, pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Regenerate Dataset", command=self.regenerate_data).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Show Statistics", command=self.show_statistics).pack(side='left', padx=5)
        
    def setup_svd_tab(self):
        """Setup SVD analysis tab"""
        frame = ttk.Frame(self.tab_svd, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', pady=5)
        
        ttk.Label(control_frame, text="Number of Components:").pack(side='left', padx=5)
        self.svd_components = ttk.Spinbox(control_frame, from_=2, to=50, width=10)
        self.svd_components.set(10)
        self.svd_components.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Run SVD", command=self.run_svd).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Results", command=lambda: self.save_results('svd')).pack(side='left', padx=5)
        
        # Results display
        self.svd_text = scrolledtext.ScrolledText(frame, height=15, width=100)
        self.svd_text.pack(fill='both', expand=True, pady=5)
        
        # Visualization area
        self.svd_plot_frame = ttk.Frame(frame)
        self.svd_plot_frame.pack(fill='both', expand=True, pady=5)
        
    def setup_nmf_tab(self):
        """Setup NMF analysis tab"""
        frame = ttk.Frame(self.tab_nmf, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', pady=5)
        
        ttk.Label(control_frame, text="Number of Components:").pack(side='left', padx=5)
        self.nmf_components = ttk.Spinbox(control_frame, from_=2, to=50, width=10)
        self.nmf_components.set(10)
        self.nmf_components.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Run NMF", command=self.run_nmf).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Results", command=lambda: self.save_results('nmf')).pack(side='left', padx=5)
        
        # Results display
        self.nmf_text = scrolledtext.ScrolledText(frame, height=15, width=100)
        self.nmf_text.pack(fill='both', expand=True, pady=5)
        
        # Visualization area
        self.nmf_plot_frame = ttk.Frame(frame)
        self.nmf_plot_frame.pack(fill='both', expand=True, pady=5)
        
    def setup_pca_tab(self):
        """Setup PCA visualization tab"""
        frame = ttk.Frame(self.tab_pca, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', pady=5)
        
        ttk.Label(control_frame, text="Number of Components:").pack(side='left', padx=5)
        self.pca_components = ttk.Spinbox(control_frame, from_=2, to=50, width=10)
        self.pca_components.set(10)
        self.pca_components.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Run PCA", command=self.run_pca).pack(side='left', padx=5)
        
        # Results display
        self.pca_text = scrolledtext.ScrolledText(frame, height=10, width=100)
        self.pca_text.pack(fill='both', pady=5)
        
        # Visualization area
        self.pca_plot_frame = ttk.Frame(frame)
        self.pca_plot_frame.pack(fill='both', expand=True, pady=5)
        
    def setup_recommend_tab(self):
        """Setup recommendations tab"""
        frame = ttk.Frame(self.tab_recommend, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', pady=5)
        
        ttk.Label(control_frame, text="User ID:").pack(side='left', padx=5)
        self.user_selector = ttk.Combobox(control_frame, width=15, state='readonly')
        self.user_selector.pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="Method:").pack(side='left', padx=5)
        self.method_selector = ttk.Combobox(control_frame, values=['SVD', 'NMF'], width=10, state='readonly')
        self.method_selector.set('SVD')
        self.method_selector.pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="# Recommendations:").pack(side='left', padx=5)
        self.n_recommendations = ttk.Spinbox(control_frame, from_=5, to=50, width=10)
        self.n_recommendations.set(10)
        self.n_recommendations.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Get Recommendations", command=self.get_recommendations).pack(side='left', padx=5)
        
        # Results display
        self.recommend_text = scrolledtext.ScrolledText(frame, height=30, width=100)
        self.recommend_text.pack(fill='both', expand=True, pady=5)
        
    def load_data(self):
        """Load or generate dataset - checks for Kaggle data first"""
        data_loaded = False
        data_source = "Unknown"
        
        # Priority 1: Check for Kaggle/external data
        if os.path.exists('kaggle_data/ratings.csv'):
            try:
                messagebox.showinfo("Loading Data", "Kaggle dataset detected. Loading external data...")
                from data_loader import ExternalDataLoader
                
                loader = ExternalDataLoader()
                
                # Try MovieLens format first
                if os.path.exists('kaggle_data/movies.csv'):
                    self.item_df, self.user_df, self.ratings_matrix = loader.load_movielens(
                        ratings_file='kaggle_data/ratings.csv',
                        movies_file='kaggle_data/movies.csv'
                    )
                    data_source = "Kaggle MovieLens"
                else:
                    # Try generic ratings format
                    self.item_df, self.user_df, self.ratings_matrix = loader.load_from_ratings_file(
                        'kaggle_data/ratings.csv',
                        user_col='userId',
                        item_col='movieId',
                        rating_col='rating'
                    )
                    data_source = "Kaggle Dataset"
                
                # Check if meets requirements
                if len(self.user_df) < 15 or len(self.item_df) < 10000:
                    response = messagebox.askyesno(
                        "Dataset Warning",
                        f"Loaded dataset has {len(self.user_df)} users and {len(self.item_df)} items.\n"
                        f"Project requires 15+ users and 10,000+ items.\n\n"
                        f"Continue with this dataset anyway?"
                    )
                    if not response:
                        messagebox.showinfo("Info", "Falling back to synthetic data generation...")
                        raise ValueError("Dataset doesn't meet requirements")
                
                # Save processed data for future use
                loader.save_processed_data('data/')
                data_loaded = True
                
            except Exception as e:
                messagebox.showwarning(
                    "Load Error", 
                    f"Failed to load Kaggle data: {str(e)}\n\nGenerating synthetic data instead..."
                )
                data_loaded = False
        
        # Priority 2: Check for processed data
        if not data_loaded and os.path.exists('data/ratings_matrix.npy'):
            try:
                self.item_df = pd.read_csv('data/courses.csv')
                self.user_df = pd.read_csv('data/students.csv')
                self.ratings_matrix = np.load('data/ratings_matrix.npy')
                data_source = "Processed Data"
                data_loaded = True
            except Exception as e:
                messagebox.showwarning("Load Error", f"Failed to load processed data: {str(e)}")
                data_loaded = False
        
        # Priority 3: Generate synthetic data
        if not data_loaded:
            messagebox.showinfo("Generating Data", "Generating synthetic dataset...")
            generator = AcademicDataGenerator(n_users=15, n_items=10000, sparsity=0.95)
            self.item_df, self.user_df, self.ratings_matrix = generator.generate_dataset()
            data_source = "Synthetic Data"
        
        # Initialize miner
        self.miner = MatrixMiner(self.ratings_matrix, self.item_df, self.user_df)
        
        # Update UI
        self.update_data_display()
        self.user_selector['values'] = self.user_df['user_id'].tolist()
        if len(self.user_df) > 0:
            self.user_selector.current(0)
        
        # Show data source in title
        self.root.title(f"Matrix Mining in Academic Ecosystems - ISU Item Recommender [{data_source}]")
        
    def regenerate_data(self):
        """Regenerate dataset"""
        if messagebox.askyesno("Confirm", "Regenerate dataset? This will overwrite existing data."):
            generator = AcademicDataGenerator(n_users=15, n_items=10000, sparsity=0.95)
            self.item_df, self.user_df, self.ratings_matrix = generator.generate_dataset()
            self.miner = MatrixMiner(self.ratings_matrix, self.item_df, self.user_df)
            self.update_data_display()
            messagebox.showinfo("Success", "Dataset regenerated successfully!")
        
    def update_data_display(self):
        """Update data overview display — works with any dataset columns."""
        self.data_text.delete(1.0, tk.END)

        # Build item statistics section dynamically
        item_stats_lines = [f"Total Items: {len(self.item_df)}"]
        for col in self.item_df.columns:
            if col == 'item_id':
                continue
            try:
                if self.item_df[col].dtype == object or self.item_df[col].nunique() < 30:
                    vc = self.item_df[col].value_counts().head(10)
                    item_stats_lines.append(f"\n{col.capitalize()} Distribution (top 10):")
                    item_stats_lines.append(vc.to_string())
                else:
                    desc = self.item_df[col].describe()
                    item_stats_lines.append(
                        f"{col.capitalize()}: min={desc['min']:.2f}, "
                        f"mean={desc['mean']:.2f}, max={desc['max']:.2f}"
                    )
            except Exception:
                pass
        item_stats_text = "\n".join(item_stats_lines)

        rated = np.count_nonzero(self.ratings_matrix)
        avg_r = self.ratings_matrix[self.ratings_matrix > 0].mean() if rated > 0 else 0
        std_r = self.ratings_matrix[self.ratings_matrix > 0].std()  if rated > 0 else 0

        info = f"""DATASET OVERVIEW
{'='*80}

Matrix Dimensions: {self.ratings_matrix.shape[0]} users x {self.ratings_matrix.shape[1]} items

USERS ({len(self.user_df)} total)
{'-'*80}
{self.user_df.to_string()}

ITEM STATISTICS
{'-'*80}
{item_stats_text}

RATING STATISTICS
{'-'*80}
Total Ratings : {rated}
Sparsity      : {(1 - rated / self.ratings_matrix.size)*100:.2f}%
Average Rating: {avg_r:.2f}
Rating Std Dev: {std_r:.2f}

Ratings per User:
{pd.Series([np.count_nonzero(self.ratings_matrix[i]) for i in range(len(self.user_df))]).describe().to_string()}
"""
        self.data_text.insert(1.0, info)
        
    def show_statistics(self):
        """Show detailed statistics — works with any dataset columns."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Dataset Statistics', fontsize=16)

        # ── Plot 1: Rating distribution ──────────────────────────────
        ratings_flat = self.ratings_matrix[self.ratings_matrix > 0]
        axes[0, 0].hist(ratings_flat, bins=20, edgecolor='black', color='#2980b9')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')

        # ── Plot 2: Ratings per user ──────────────────────────────────
        ratings_per_user = [
            np.count_nonzero(self.ratings_matrix[i])
            for i in range(len(self.user_df))
        ]
        axes[0, 1].bar(range(len(ratings_per_user)), ratings_per_user, color='#27ae60')
        axes[0, 1].set_title('Ratings per User')
        axes[0, 1].set_xlabel('User Index')
        axes[0, 1].set_ylabel('Number of Ratings')

        # ── Plot 3: First categorical column (any name) ───────────────
        cat_col = next(
            (c for c in self.item_df.columns
             if c != 'item_id' and (
                 self.item_df[c].dtype == object or
                 self.item_df[c].nunique() < 30
             )),
            None
        )
        if cat_col:
            counts = self.item_df[cat_col].value_counts().head(15)
            axes[1, 0].bar(counts.index.astype(str), counts.values, color='#8e44ad')
            axes[1, 0].set_title(f'Items by {cat_col.capitalize()}')
            axes[1, 0].set_xlabel(cat_col.capitalize())
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No categorical column found',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Category Distribution')

        # ── Plot 4: First numeric column (any name) ───────────────────
        num_col = next(
            (c for c in self.item_df.columns
             if c != 'item_id' and
             pd.api.types.is_numeric_dtype(self.item_df[c]) and
             self.item_df[c].nunique() >= 3),
            None
        )
        if num_col:
            axes[1, 1].hist(
                self.item_df[num_col].dropna(), bins=15,
                edgecolor='black', color='#e67e22'
            )
            axes[1, 1].set_title(f'Item {num_col.capitalize()} Distribution')
            axes[1, 1].set_xlabel(num_col.capitalize())
            axes[1, 1].set_ylabel('Frequency')
        else:
            axes[1, 1].text(0.5, 0.5, 'No numeric column found',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Numeric Distribution')

        plt.tight_layout()

        # Save figure
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/dataset_statistics.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def run_svd(self):
        """Run SVD analysis"""
        try:
            n_components = int(self.svd_components.get())
            self.svd_results = self.miner.apply_svd(n_components=n_components)
            
            # Display results
            self.svd_text.delete(1.0, tk.END)
            results_text = f"""SVD ANALYSIS RESULTS
{'='*80}

Number of Components: {self.svd_results['n_components']}
Total Variance Explained: {self.svd_results['total_variance_explained']*100:.2f}%
Reconstruction RMSE: {self.svd_results['rmse']:.4f}

EXPLAINED VARIANCE BY COMPONENT:
{'-'*80}
"""
            for i, (var, cum_var) in enumerate(zip(self.svd_results['explained_variance_ratio'], 
                                                     self.svd_results['cumulative_variance'])):
                results_text += f"Component {i+1:2d}: {var*100:6.2f}% (Cumulative: {cum_var*100:6.2f}%)\n"
            
            results_text += f"\nSINGULAR VALUES:\n{'-'*80}\n"
            for i, sv in enumerate(self.svd_results['singular_values'][:10]):
                results_text += f"σ{i+1:2d} = {sv:.2f}\n"
            
            self.svd_text.insert(1.0, results_text)
            
            # Plot results
            self.plot_svd_results()
            
            messagebox.showinfo("Success", "SVD analysis completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"SVD analysis failed: {str(e)}")
    
    def plot_svd_results(self):
        """Plot SVD analysis results"""
        # Clear previous plots
        for widget in self.svd_plot_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Explained variance
        axes[0].plot(range(1, len(self.svd_results['explained_variance_ratio'])+1),
                     self.svd_results['cumulative_variance'], 'bo-')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('SVD: Explained Variance')
        axes[0].grid(True, alpha=0.3)
        
        # Singular values
        axes[1].plot(range(1, len(self.svd_results['singular_values'])+1),
                     self.svd_results['singular_values'], 'ro-')
        axes[1].set_xlabel('Component')
        axes[1].set_ylabel('Singular Value')
        axes[1].set_title('SVD: Singular Values')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.svd_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def run_nmf(self):
        """Run NMF analysis"""
        try:
            n_components = int(self.nmf_components.get())
            self.nmf_results = self.miner.apply_nmf(n_components=n_components)
            
            # Display results
            self.nmf_text.delete(1.0, tk.END)
            results_text = f"""NMF ANALYSIS RESULTS
{'='*80}

Number of Components: {self.nmf_results['n_components']}
Reconstruction Error: {self.nmf_results['reconstruction_error']:.4f}
Reconstruction RMSE: {self.nmf_results['rmse']:.4f}
Iterations: {self.nmf_results['n_iter']}

LATENT FACTORS:
{'-'*80}
User Factors Shape: {self.nmf_results['user_factors'].shape}
Item Factors Shape: {self.nmf_results['item_factors'].shape}

FACTOR STATISTICS:
{'-'*80}
User Factors - Mean: {self.nmf_results['user_factors'].mean():.4f}, Std: {self.nmf_results['user_factors'].std():.4f}
Item Factors - Mean: {self.nmf_results['item_factors'].mean():.4f}, Std: {self.nmf_results['item_factors'].std():.4f}
"""
            
            self.nmf_text.insert(1.0, results_text)
            
            # Plot results
            self.plot_nmf_results()
            
            messagebox.showinfo("Success", "NMF analysis completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"NMF analysis failed: {str(e)}")
    
    def plot_nmf_results(self):
        """Plot NMF analysis results"""
        # Clear previous plots
        for widget in self.nmf_plot_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # User factors heatmap
        sns.heatmap(self.nmf_results['user_factors'][:10], ax=axes[0], cmap='YlOrRd')
        axes[0].set_title('NMF: User Latent Factors (First 10 Students)')
        axes[0].set_xlabel('Factor')
        axes[0].set_ylabel('User')
        
        # Item factors distribution
        axes[1].boxplot([self.nmf_results['item_factors'][:, i] 
                         for i in range(min(10, self.nmf_results['n_components']))],
                        labels=[f'F{i+1}' for i in range(min(10, self.nmf_results['n_components']))])
        axes[1].set_title('NMF: Item Factor Distributions')
        axes[1].set_xlabel('Factor')
        axes[1].set_ylabel('Factor Value')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.nmf_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def run_pca(self):
        """Run PCA analysis"""
        try:
            n_components = int(self.pca_components.get())
            self.pca_results = self.miner.apply_pca(n_components=n_components)
            
            # Display results
            self.pca_text.delete(1.0, tk.END)
            results_text = f"""PCA ANALYSIS RESULTS
{'='*80}

Number of Components: {self.pca_results['n_components']}
Total Variance Explained: {self.pca_results['total_variance_explained']*100:.2f}%

EXPLAINED VARIANCE BY COMPONENT:
{'-'*80}
"""
            for i, (var, cum_var) in enumerate(zip(self.pca_results['explained_variance_ratio'], 
                                                     self.pca_results['cumulative_variance'])):
                results_text += f"PC{i+1:2d}: {var*100:6.2f}% (Cumulative: {cum_var*100:6.2f}%)\n"
            
            self.pca_text.insert(1.0, results_text)
            
            # Plot results
            self.plot_pca_results()
            
            messagebox.showinfo("Success", "PCA analysis completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"PCA analysis failed: {str(e)}")
    
    def plot_pca_results(self):
        """Plot PCA analysis results"""
        # Clear previous plots
        for widget in self.pca_plot_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Scree plot
        axes[0].plot(range(1, len(self.pca_results['explained_variance_ratio'])+1),
                     self.pca_results['explained_variance_ratio'], 'bo-')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA: Scree Plot')
        axes[0].grid(True, alpha=0.3)
        
        # 2D projection
        if self.pca_results['n_components'] >= 2:
            scatter = axes[1].scatter(self.pca_results['transformed_data'][:, 0],
                                     self.pca_results['transformed_data'][:, 1],
                                     c=range(len(self.user_df)), cmap='viridis', s=100)
            axes[1].set_xlabel('PC1')
            axes[1].set_ylabel('PC2')
            axes[1].set_title('PCA: User Projection (PC1 vs PC2)')
            plt.colorbar(scatter, ax=axes[1], label='User Index')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.pca_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def get_recommendations(self):
        """Get item recommendations for selected user (Universal Version)"""
        try:
            user_id = self.user_selector.get()

            # Match the dtype of user_id column for reliable comparison
            uid_dtype = self.user_df['user_id'].dtype
            try:
                if np.issubdtype(uid_dtype, np.integer):
                    search_id = int(user_id)
                elif np.issubdtype(uid_dtype, np.floating):
                    search_id = float(user_id)
                else:
                    search_id = str(user_id)
            except (ValueError, TypeError):
                search_id = user_id

            # Use positional index (needed for matrix row lookup)
            matches = np.where(self.user_df['user_id'].values == search_id)[0]
            if len(matches) == 0:
                messagebox.showerror("Error", f"User ID '{user_id}' not found in dataset.")
                return
            user_idx = int(matches[0])
            method = self.method_selector.get().lower()
            n_rec = int(self.n_recommendations.get())
            
            # Get recommendations
            if method == 'svd':
                if self.svd_results is None:
                    messagebox.showwarning("Warning", "Please run SVD analysis first!")
                    return
                recommendations = self.miner.recommend_items_svd(user_idx, n_rec)
            else:
                if self.nmf_results is None:
                    messagebox.showwarning("Warning", "Please run NMF analysis first!")
                    return
                recommendations = self.miner.recommend_items_nmf(user_idx, n_rec)
            
            # Display recommendations
            self.recommend_text.delete(1.0, tk.END)
            
            # Dynamically print User Info (skipping the internal user_id column)
            user_info = self.user_df.iloc[user_idx]
            results_text = f"""ITEM RECOMMENDATIONS
{'='*80}

USER PROFILE:
{'-'*80}
User ID: {user_id}\n"""
            
            for col in self.user_df.columns:
                if col != 'user_id':
                    results_text += f"{col.capitalize()}: {user_info[col]}\n"

            results_text += f"""
Method: {method.upper()}
Number of Recommendations: {n_rec}

RECOMMENDED ITEMS:
{'-'*80}
"""
            
            # Dynamically print Item Info
            for i, row in recommendations.iterrows():
                results_text += f"\n{i+1}. Item ID: {row['item_id']}\n"
                results_text += f"   Predicted Rating: {row['predicted_rating']:.2f}\n"
                
                # Print all other metadata columns dynamically
                for col in recommendations.columns:
                    if col not in ['item_id', 'predicted_rating', 'actual_rating']:
                        val = row[col]
                        # Format floats nicely, leave strings alone
                        if isinstance(val, float):
                            results_text += f"   {col.capitalize()}: {val:.2f}\n"
                        else:
                            results_text += f"   {col.capitalize()}: {val}\n"
            
            # Show user's current ratings
            rated_items = np.where(self.ratings_matrix[user_idx] > 0)[0]
            results_text += f"\n\nUSER'S CURRENT RATINGS ({len(rated_items)} items):\n{'-'*80}\n"
            for idx in rated_items[:10]:  # Show first 10
                item = self.item_df.iloc[idx]
                rating = self.ratings_matrix[user_idx, idx]
                results_text += f"Item {item['item_id']}: {rating:.1f} stars\n"
            
            if len(rated_items) > 10:
                results_text += f"... and {len(rated_items)-10} more items\n"
            
            self.recommend_text.insert(1.0, results_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Recommendation failed: {str(e)}")
    
    def save_results(self, method):
        """Save analysis results to file"""
        try:
            os.makedirs('results', exist_ok=True)
            
            if method == 'svd' and self.svd_results:
                # Save SVD results
                np.save('results/svd_student_factors.npy', self.svd_results['user_factors'])
                np.save('results/svd_course_factors.npy', self.svd_results['item_factors'])
                
                # Save metrics
                with open('results/svd_metrics.txt', 'w') as f:
                    f.write(f"SVD Analysis Results\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Components: {self.svd_results['n_components']}\n")
                    f.write(f"Variance Explained: {self.svd_results['total_variance_explained']*100:.2f}%\n")
                    f.write(f"RMSE: {self.svd_results['rmse']:.4f}\n")
                
                messagebox.showinfo("Success", "SVD results saved to results/ directory")
                
            elif method == 'nmf' and self.nmf_results:
                # Save NMF results
                np.save('results/nmf_student_factors.npy', self.nmf_results['user_factors'])
                np.save('results/nmf_course_factors.npy', self.nmf_results['item_factors'])
                
                # Save metrics
                with open('results/nmf_metrics.txt', 'w') as f:
                    f.write(f"NMF Analysis Results\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Components: {self.nmf_results['n_components']}\n")
                    f.write(f"Reconstruction Error: {self.nmf_results['reconstruction_error']:.4f}\n")
                    f.write(f"RMSE: {self.nmf_results['rmse']:.4f}\n")
                
                messagebox.showinfo("Success", "NMF results saved to results/ directory")
            else:
                messagebox.showwarning("Warning", f"No {method.upper()} results to save. Run analysis first!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def load_movielens_dialog(self):
        """Dialog to load MovieLens dataset with separate ratings and movies CSV files"""
        from tkinter import filedialog

        # --- Step 1: Build the file-picker dialog ---
        dialog = tk.Toplevel(self.root)
        dialog.title("Load MovieLens Dataset")
        dialog.geometry("560x280")
        dialog.resizable(False, False)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text="Load MovieLens Dataset",
            font=('Arial', 13, 'bold')
        ).pack(pady=(14, 2))
        ttk.Label(
            dialog,
            text="Select your ratings CSV (required) and movies CSV (optional for metadata).",
            foreground='gray',
            wraplength=520
        ).pack(pady=(0, 10))

        frame = ttk.Frame(dialog, padding="12")
        frame.pack(fill='both', expand=True)
        frame.columnconfigure(1, weight=1)

        ratings_var = tk.StringVar()
        movies_var  = tk.StringVar()

        # Pre-fill if kaggle_data folder exists
        if os.path.exists('kaggle_data/ratings.csv'):
            ratings_var.set(os.path.abspath('kaggle_data/ratings.csv'))
        if os.path.exists('kaggle_data/movies.csv'):
            movies_var.set(os.path.abspath('kaggle_data/movies.csv'))

        def browse_ratings():
            path = filedialog.askopenfilename(
                title="Select Ratings CSV",
                initialdir=os.path.dirname(ratings_var.get()) if ratings_var.get() else '.',
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if path:
                ratings_var.set(path)

        def browse_movies():
            path = filedialog.askopenfilename(
                title="Select Movies / Items CSV (optional)",
                initialdir=os.path.dirname(movies_var.get()) if movies_var.get() else '.',
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if path:
                movies_var.set(path)

        # Ratings row
        ttk.Label(frame, text="Ratings CSV  *", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, sticky='w', pady=6, padx=(0, 8))
        ttk.Entry(frame, textvariable=ratings_var, width=42).grid(
            row=0, column=1, sticky='ew', pady=6)
        ttk.Button(frame, text="Browse…", command=browse_ratings).grid(
            row=0, column=2, padx=(6, 0), pady=6)

        # Movies row
        ttk.Label(frame, text="Movies CSV", font=('Arial', 9)).grid(
            row=1, column=0, sticky='w', pady=6, padx=(0, 8))
        ttk.Entry(frame, textvariable=movies_var, width=42).grid(
            row=1, column=1, sticky='ew', pady=6)
        ttk.Button(frame, text="Browse…", command=browse_movies).grid(
            row=1, column=2, padx=(6, 0), pady=6)

        result = {'confirmed': False}

        def confirm():
            if not ratings_var.get():
                messagebox.showwarning("Missing File", "Please select a Ratings CSV file.",
                                       parent=dialog)
                return
            result['confirmed']    = True
            result['ratings_file'] = ratings_var.get()
            result['movies_file']  = movies_var.get() or None
            dialog.destroy()

        def cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="  Load Dataset  ", command=confirm,
                  bg='#2980b9', fg='white', font=('Arial', 10, 'bold'),
                  relief='flat', padx=8, pady=4, cursor='hand2').pack(side='left', padx=6)
        ttk.Button(btn_frame, text="Cancel", command=cancel).pack(side='left', padx=6)

        dialog.wait_window()

        if not result['confirmed']:
            return

        # --- Step 2: Load the files ---
        try:
            from data_loader import ExternalDataLoader
            loader = ExternalDataLoader()

            self.item_df, self.user_df, self.ratings_matrix = loader.load_movielens(
                ratings_file=result['ratings_file'],
                movies_file=result['movies_file']
            )

            if len(self.user_df) < 15 or len(self.item_df) < 10000:
                response = messagebox.askyesno(
                    "Dataset Warning",
                    f"Loaded dataset has {len(self.user_df)} users and {len(self.item_df)} items.\n"
                    f"Project requires 15+ users and 10,000+ items.\n\nContinue anyway?"
                )
                if not response:
                    return

            loader.save_processed_data('data/')
            self.miner = MatrixMiner(self.ratings_matrix, self.item_df, self.user_df)
            self.svd_results = None
            self.nmf_results = None
            self.pca_results = None
            self.update_data_display()
            self.user_selector['values'] = self.user_df['user_id'].tolist()
            if len(self.user_df) > 0:
                self.user_selector.current(0)

            movies_label = os.path.basename(result['movies_file']) if result['movies_file'] else 'none'
            self.root.title("Matrix Mining - ISU Item Recommender [MovieLens]")
            messagebox.showinfo(
                "Loaded",
                f"MovieLens dataset loaded!\n\n"
                f"Users  : {len(self.user_df)}\n"
                f"Items  : {len(self.item_df)}\n"
                f"Ratings: {int((self.ratings_matrix > 0).sum())}\n"
                f"Movies file: {movies_label}"
            )

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load MovieLens data:\n{str(e)}")

    def load_kaggle_dataset(self):
        """Load a single ratings CSV (generic format)"""
        from tkinter import filedialog

        filepath = filedialog.askopenfilename(
            title="Select Ratings CSV File",
            initialdir="kaggle_data" if os.path.exists("kaggle_data") else ".",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            from data_loader import ExternalDataLoader
            loader = ExternalDataLoader()

            sample_df = pd.read_csv(filepath, nrows=0, encoding='utf-8-sig')
            detected_cols = list(sample_df.columns)
            if not detected_cols:
                messagebox.showerror("Error", "Could not read column headers.")
                return

            def _best_match(keywords, columns):
                for kw in keywords:
                    for col in columns:
                        if kw.lower() in col.lower():
                            return col
                return columns[0] if columns else ""

            default_user   = _best_match(["user", "uid"], detected_cols)
            default_item   = _best_match(["item", "movie", "product", "book"], detected_cols)
            default_rating = _best_match(["rating", "score", "grade", "rate"], detected_cols)

            dialog = tk.Toplevel(self.root)
            dialog.title("Column Configuration")
            dialog.geometry("460x280")
            dialog.resizable(False, False)
            dialog.grab_set()

            ttk.Label(dialog, text="Map CSV columns to User / Item / Rating:",
                      font=('Arial', 10, 'bold'), wraplength=420).pack(pady=12, padx=10)
            frame = ttk.Frame(dialog, padding="10")
            frame.pack(fill='both', expand=True)

            ttk.Label(frame, text="Available: " + ", ".join(detected_cols),
                      foreground='gray', wraplength=400).grid(
                row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))

            ttk.Label(frame, text="User Column:").grid(row=1, column=0, sticky='w', pady=5, padx=5)
            user_col = ttk.Combobox(frame, values=detected_cols, width=22, state='readonly')
            user_col.set(default_user)
            user_col.grid(row=1, column=1, pady=5, padx=5)

            ttk.Label(frame, text="Item Column:").grid(row=2, column=0, sticky='w', pady=5, padx=5)
            item_col = ttk.Combobox(frame, values=detected_cols, width=22, state='readonly')
            item_col.set(default_item)
            item_col.grid(row=2, column=1, pady=5, padx=5)

            ttk.Label(frame, text="Rating Column:").grid(row=3, column=0, sticky='w', pady=5, padx=5)
            rating_col = ttk.Combobox(frame, values=detected_cols, width=22, state='readonly')
            rating_col.set(default_rating)
            rating_col.grid(row=3, column=1, pady=5, padx=5)

            result = {'confirmed': False}

            def confirm():
                result['confirmed']  = True
                result['user_col']   = user_col.get()
                result['item_col']   = item_col.get()
                result['rating_col'] = rating_col.get()
                dialog.destroy()

            def cancel():
                dialog.destroy()

            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=10)
            ttk.Button(btn_frame, text="Load", command=confirm).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="Cancel", command=cancel).pack(side='left', padx=5)
            dialog.wait_window()

            if not result['confirmed']:
                return

            self.item_df, self.user_df, self.ratings_matrix = loader.load_from_ratings_file(
                filepath,
                user_col=result['user_col'],
                item_col=result['item_col'],
                rating_col=result['rating_col']
            )

            if len(self.user_df) < 15 or len(self.item_df) < 10000:
                response = messagebox.askyesno(
                    "Dataset Warning",
                    f"Loaded dataset has {len(self.user_df)} users and {len(self.item_df)} items.\n"
                    f"Project requires 15+ users and 10,000+ items.\n\nContinue anyway?"
                )
                if not response:
                    return

            loader.save_processed_data('data/')
            self.miner = MatrixMiner(self.ratings_matrix, self.item_df, self.user_df)
            self.svd_results = None
            self.nmf_results = None
            self.pca_results = None
            self.update_data_display()
            self.user_selector['values'] = self.user_df['user_id'].tolist()
            if len(self.user_df) > 0:
                self.user_selector.current(0)
            self.root.title("Matrix Mining - ISU Item Recommender [Custom Dataset]")
            messagebox.showinfo("Success",
                f"Dataset loaded!\nUsers: {len(self.user_df)}  Items: {len(self.item_df)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")

    def load_custom_csv(self):
        """Load custom CSV dataset"""
        self.load_kaggle_dataset()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Matrix Mining in Academic Ecosystems
Version 1.0

Applying SVD and NMF to Item Collaborative Filtering at ISU

Features:
• SVD, NMF, and PCA algorithms
• Support for synthetic and custom CSV datasets
• Real-time analysis and visualization

Built for ISU Data Mining Item
Topic 2: Mining Matrix Data"""

        messagebox.showinfo("About", about_text)

    def close_app(self):
        """Confirm and close the application"""
        if messagebox.askyesno("Close Application", "Are you sure you want to close the application?"):
            self.root.destroy()

def main():
    root = tk.Tk()
    app = MatrixMiningApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
