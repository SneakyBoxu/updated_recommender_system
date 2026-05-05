"""
MineMatrix Flask Web Application
Matrix Mining in Academic Ecosystems — Web Interface with Real-Time Processing
"""

import os
import io
import sys
import json
import queue
import threading
import base64
import time

# Force UTF-8 stdout/stderr on Windows to avoid cp1252 encoding errors
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, Response, jsonify, stream_with_context

from data_generator import AcademicDataGenerator
from data_loader import ExternalDataLoader
from matrix_mining import MatrixMiner

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'minematrix-flask-secret'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

# ---------------------------------------------------------------------------
# Module-level state (single-user local prototype)
# ---------------------------------------------------------------------------
state = {
    'item_df': None,
    'user_df': None,
    'ratings_matrix': None,
    'miner': None,
    'svd_results': None,
    'nmf_results': None,
    'pca_results': None,
    'data_source': 'Not loaded',
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_application_data():
    """Load data using the same priority chain as the Tkinter app."""
    data_loaded = False
    data_source = 'Unknown'

    # Priority 1: Kaggle data
    if os.path.exists('kaggle_data/ratings.csv'):
        try:
            loader = ExternalDataLoader()
            if os.path.exists('kaggle_data/movies.csv'):
                state['item_df'], state['user_df'], state['ratings_matrix'] = loader.load_movielens(
                    ratings_file='kaggle_data/ratings.csv',
                    movies_file='kaggle_data/movies.csv'
                )
                data_source = 'Kaggle MovieLens'
            else:
                state['item_df'], state['user_df'], state['ratings_matrix'] = loader.load_from_ratings_file(
                    'kaggle_data/ratings.csv', user_col='userId', item_col='movieId', rating_col='rating'
                )
                data_source = 'Kaggle Dataset'
            loader.save_processed_data('data/')
            data_loaded = True
        except Exception as e:
            print(f'[WARN] Could not load Kaggle data: {e}')

    # Priority 2: Pre-processed data
    if not data_loaded and os.path.exists('data/ratings_matrix.npy'):
        try:
            state['item_df'] = pd.read_csv('data/courses.csv')
            state['user_df'] = pd.read_csv('data/students.csv')
            state['ratings_matrix'] = np.load('data/ratings_matrix.npy')
            data_source = 'Processed Data'
            data_loaded = True
        except Exception as e:
            print(f'[WARN] Could not load processed data: {e}')

    # Priority 3: Synthetic data
    if not data_loaded:
        print('[INFO] Generating synthetic dataset (15 users × 10 000 items)…')
        generator = AcademicDataGenerator(n_users=15, n_items=10000, sparsity=0.95)
        state['item_df'], state['user_df'], state['ratings_matrix'] = generator.generate_dataset()
        data_source = 'Synthetic Data'

    state['miner'] = MatrixMiner(state['ratings_matrix'], state['item_df'], state['user_df'])
    state['data_source'] = data_source
    print(f'[INFO] Data loaded: {data_source}')


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------
def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{encoded}'


def dark_fig(nrows=1, ncols=1, figsize=(12, 4)):
    """Create a matplotlib figure with a dark background theme."""
    bg = '#0d1117'
    text_color = '#c9d1d9'
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=bg)
    axes_flat = [axes] if (nrows == 1 and ncols == 1) else \
                (axes.flatten() if hasattr(axes, 'flatten') else [axes])
    for ax in axes_flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
    return fig, axes


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------
def sse_event(data_dict):
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data_dict)}\n\n"


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Dashboard / Data Overview."""
    if state['item_df'] is None:
        load_application_data()

    rm = state['ratings_matrix']
    rated = int(np.count_nonzero(rm))
    sparsity = (1 - rated / rm.size) * 100
    avg_rating = float(rm[rm > 0].mean()) if rated > 0 else 0

    stats = {
        'n_users': int(rm.shape[0]),
        'n_items': int(rm.shape[1]),
        'total_ratings': rated,
        'sparsity': round(sparsity, 2),
        'avg_rating': round(avg_rating, 2),
        'data_source': state['data_source'],
    }
    return render_template('index.html', stats=stats,
                           users=state['user_df'].to_dict('records'),
                           items_sample=state['item_df'].head(20).to_dict('records'),
                           item_cols=[c for c in state['item_df'].columns])


@app.route('/svd')
def svd_page():
    return render_template('svd.html',
                           svd_results=_serialize_results('svd'),
                           data_source=state['data_source'])


@app.route('/nmf')
def nmf_page():
    return render_template('nmf.html',
                           nmf_results=_serialize_results('nmf'),
                           data_source=state['data_source'])


@app.route('/pca')
def pca_page():
    return render_template('pca.html',
                           pca_results=_serialize_results('pca'),
                           data_source=state['data_source'])


@app.route('/recommend')
def recommend_page():
    users = state['user_df']['user_id'].tolist() if state['user_df'] is not None else []
    return render_template('recommend.html', users=users,
                           data_source=state['data_source'])


@app.route('/upload')
def upload_page():
    """Load Dataset page."""
    rm = state['ratings_matrix']
    current_stats = None
    if rm is not None:
        rated = int(np.count_nonzero(rm))
        current_stats = {
            'n_users': int(rm.shape[0]),
            'n_items': int(rm.shape[1]),
            'total_ratings': rated,
            'sparsity': round((1 - rated / rm.size) * 100, 2),
        }
    return render_template('upload.html',
                           current_source=state['data_source'],
                           current_stats=current_stats,
                           data_source=state['data_source'])


@app.route('/upload', methods=['POST'])
def handle_upload():
    """Accept ratings.csv (required) + movies.csv (optional) and load them."""
    if 'ratings_file' not in request.files or request.files['ratings_file'].filename == '':
        return jsonify({'ok': False, 'error': 'ratings_file is required'}), 400

    ratings_file = request.files['ratings_file']
    movies_file  = request.files.get('movies_file')

    max_users   = int(request.form.get('max_users', 0))
    min_ratings = int(request.form.get('min_ratings', 5))

    # Save uploads to a temp folder inside the project
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle_data')
    os.makedirs(upload_dir, exist_ok=True)

    ratings_path = os.path.join(upload_dir, 'ratings.csv')
    ratings_file.save(ratings_path)

    movies_path = None
    if movies_file and movies_file.filename != '':
        movies_path = os.path.join(upload_dir, 'movies.csv')
        movies_file.save(movies_path)

    try:
        loader = ExternalDataLoader()

        if movies_path:
            item_df, user_df, ratings_matrix = loader.load_movielens(
                ratings_file=ratings_path,
                movies_file=movies_path
            )
            data_source = 'MovieLens (with metadata)'
        else:
            # Auto-detect column names
            sample = pd.read_csv(ratings_path, nrows=1)
            cols = sample.columns.tolist()
            user_col   = next((c for c in ['userId', 'user_id', 'user'] if c in cols), cols[0])
            item_col   = next((c for c in ['movieId', 'item_id', 'itemId', 'movie'] if c in cols), cols[1])
            rating_col = next((c for c in ['rating', 'score', 'rate'] if c in cols), cols[2])
            item_df, user_df, ratings_matrix = loader.load_from_ratings_file(
                ratings_path, user_col=user_col, item_col=item_col, rating_col=rating_col
            )
            data_source = 'MovieLens (ratings only)'

        # Optional: filter users with too few ratings
        if min_ratings > 1 and hasattr(loader, 'filter_data'):
            loader.filter_data(min_ratings_per_user=min_ratings, min_ratings_per_item=1)
            item_df      = loader.item_df
            user_df      = loader.user_df
            ratings_matrix = loader.ratings_matrix

        # Optional: cap number of users (keeps first N)
        if max_users > 0 and len(user_df) > max_users:
            user_df        = user_df.iloc[:max_users].reset_index(drop=True)
            ratings_matrix = ratings_matrix[:max_users, :]

        # Update global state
        state['item_df']        = item_df
        state['user_df']        = user_df
        state['ratings_matrix'] = ratings_matrix
        state['miner']          = MatrixMiner(ratings_matrix, item_df, user_df)
        state['svd_results']    = None
        state['nmf_results']    = None
        state['pca_results']    = None
        state['data_source']    = data_source

        # Persist processed data
        loader.item_df        = item_df
        loader.user_df        = user_df
        loader.ratings_matrix = ratings_matrix
        loader.save_processed_data('data/')

        rated = int(np.count_nonzero(ratings_matrix))
        return jsonify({
            'ok': True,
            'data_source': data_source,
            'n_users':   int(ratings_matrix.shape[0]),
            'n_items':   int(ratings_matrix.shape[1]),
            'n_ratings': rated,
            'sparsity':  round((1 - rated / ratings_matrix.size) * 100, 2),
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — SSE Analysis Streams
# ---------------------------------------------------------------------------
@app.route('/run/svd')
def run_svd_stream():
    """Stream SVD computation via SSE."""
    n_components = int(request.args.get('n', 10))

    def generate():
        q = queue.Queue()

        def callback(msg, level='info'):
            q.put({'type': 'log', 'msg': msg, 'level': level})

        def worker():
            try:
                results = state['miner'].apply_svd(
                    n_components=n_components, log_callback=callback
                )
                # Build summary payload (numpy types → python)
                summary = {
                    'n_components': int(results['n_components']),
                    'total_variance': round(float(results['total_variance_explained']) * 100, 2),
                    'rmse': round(float(results['rmse']), 4),
                    'singular_values': [round(float(v), 2) for v in results['singular_values']],
                    'explained_variance': [round(float(v) * 100, 2) for v in results['explained_variance_ratio']],
                    'cumulative_variance': [round(float(v) * 100, 2) for v in results['cumulative_variance']],
                }
                state['svd_results'] = results
                q.put({'type': 'done', 'results': summary})
            except Exception as e:
                q.put({'type': 'error', 'msg': str(e)})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            try:
                item = q.get(timeout=60)
                yield sse_event(item)
                if item['type'] in ('done', 'error'):
                    break
            except queue.Empty:
                yield sse_event({'type': 'error', 'msg': 'Timeout waiting for SVD'})
                break

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/run/nmf')
def run_nmf_stream():
    """Stream NMF computation via SSE."""
    n_components = int(request.args.get('n', 10))

    def generate():
        q = queue.Queue()

        def callback(msg, level='info'):
            q.put({'type': 'log', 'msg': msg, 'level': level})

        def worker():
            try:
                results = state['miner'].apply_nmf(
                    n_components=n_components, log_callback=callback
                )
                summary = {
                    'n_components': int(results['n_components']),
                    'reconstruction_error': round(float(results['reconstruction_error']), 4),
                    'rmse': round(float(results['rmse']), 4),
                    'n_iter': int(results['n_iter']),
                }
                state['nmf_results'] = results
                q.put({'type': 'done', 'results': summary})
            except Exception as e:
                q.put({'type': 'error', 'msg': str(e)})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            try:
                item = q.get(timeout=120)
                yield sse_event(item)
                if item['type'] in ('done', 'error'):
                    break
            except queue.Empty:
                yield sse_event({'type': 'error', 'msg': 'Timeout waiting for NMF'})
                break

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/run/pca')
def run_pca_stream():
    """Stream PCA computation via SSE."""
    n_components = int(request.args.get('n', 10))

    def generate():
        q = queue.Queue()

        def callback(msg, level='info'):
            q.put({'type': 'log', 'msg': msg, 'level': level})

        def worker():
            try:
                results = state['miner'].apply_pca(
                    n_components=n_components, log_callback=callback
                )
                summary = {
                    'n_components': int(results['n_components']),
                    'total_variance': round(float(results['total_variance_explained']) * 100, 2),
                    'explained_variance': [round(float(v) * 100, 2) for v in results['explained_variance_ratio']],
                    'cumulative_variance': [round(float(v) * 100, 2) for v in results['cumulative_variance']],
                }
                state['pca_results'] = results
                q.put({'type': 'done', 'results': summary})
            except Exception as e:
                q.put({'type': 'error', 'msg': str(e)})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            try:
                item = q.get(timeout=60)
                yield sse_event(item)
                if item['type'] in ('done', 'error'):
                    break
            except queue.Empty:
                yield sse_event({'type': 'error', 'msg': 'Timeout waiting for PCA'})
                break

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---------------------------------------------------------------------------
# Routes — Charts
# ---------------------------------------------------------------------------
@app.route('/api/chart/svd')
def chart_svd():
    if state['svd_results'] is None:
        return jsonify({'error': 'Run SVD first'}), 400

    res = state['svd_results']
    fig, axes = dark_fig(1, 2, figsize=(12, 4))

    ax0, ax1 = axes
    x = range(1, len(res['explained_variance_ratio']) + 1)

    ax0.plot(x, res['cumulative_variance'] * 100, 'o-', color='#6c63ff', linewidth=2)
    ax0.fill_between(x, res['cumulative_variance'] * 100, alpha=0.15, color='#6c63ff')
    ax0.set_xlabel('Components')
    ax0.set_ylabel('Cumulative Variance (%)')
    ax0.set_title('SVD — Explained Variance')
    ax0.grid(True, alpha=0.2, color='#30363d')

    ax1.bar(x, res['singular_values'], color='#2dd4bf', alpha=0.85)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('SVD — Singular Values')
    ax1.grid(True, alpha=0.2, color='#30363d', axis='y')

    plt.tight_layout()
    return jsonify({'image': fig_to_base64(fig)})


@app.route('/api/chart/nmf')
def chart_nmf():
    if state['nmf_results'] is None:
        return jsonify({'error': 'Run NMF first'}), 400

    res = state['nmf_results']
    fig, axes = dark_fig(1, 2, figsize=(12, 4))
    ax0, ax1 = axes

    user_factors = res['user_factors']
    n_show = min(10, user_factors.shape[0])
    n_factors = min(10, res['n_components'])

    text_color = '#c9d1d9'
    sns.heatmap(user_factors[:n_show, :n_factors], ax=ax0, cmap='magma',
                linewidths=0.3, linecolor='#0d1117',
                cbar_kws={'shrink': 0.8})
    ax0.set_title('NMF — User Latent Factors', color=text_color)
    ax0.set_xlabel('Factor', color=text_color)
    ax0.set_ylabel('User Index', color=text_color)
    ax0.tick_params(colors=text_color)

    item_factors = res['item_factors']
    data_bp = [item_factors[:, i] for i in range(n_factors)]
    bp = ax1.boxplot(data_bp, labels=[f'F{i+1}' for i in range(n_factors)],
                     patch_artist=True, boxprops=dict(facecolor='#6c63ff', alpha=0.6),
                     medianprops=dict(color='#2dd4bf', linewidth=2),
                     whiskerprops=dict(color='#c9d1d9'),
                     capprops=dict(color='#c9d1d9'),
                     flierprops=dict(markerfacecolor='#f97316', marker='o', markersize=3))
    ax1.set_title('NMF — Item Factor Distributions', color=text_color)
    ax1.set_xlabel('Factor', color=text_color)
    ax1.set_ylabel('Value', color=text_color)
    ax1.grid(True, alpha=0.2, color='#30363d', axis='y')

    plt.tight_layout()
    return jsonify({'image': fig_to_base64(fig)})


@app.route('/api/chart/pca')
def chart_pca():
    if state['pca_results'] is None:
        return jsonify({'error': 'Run PCA first'}), 400

    res = state['pca_results']
    text_color = '#c9d1d9'

    if res['n_components'] >= 2:
        fig, axes = dark_fig(1, 2, figsize=(12, 4))
        ax0, ax1 = axes
    else:
        fig, ax0 = dark_fig(1, 1, figsize=(6, 4))
        ax1 = None

    x = range(1, len(res['explained_variance_ratio']) + 1)
    ax0.bar(x, res['explained_variance_ratio'] * 100, color='#f97316', alpha=0.8)
    ax0.plot(x, res['cumulative_variance'] * 100, 'o-', color='#2dd4bf', linewidth=2, label='Cumulative')
    ax0.set_xlabel('Principal Component')
    ax0.set_ylabel('Variance (%)')
    ax0.set_title('PCA — Scree Plot')
    ax0.legend(labelcolor=text_color, framealpha=0.1)
    ax0.grid(True, alpha=0.2, color='#30363d', axis='y')

    if ax1 is not None:
        td = res['transformed_data']
        n_pts = td.shape[0]
        sc = ax1.scatter(td[:, 0], td[:, 1],
                         c=range(n_pts), cmap='plasma', s=120,
                         edgecolors='#c9d1d9', linewidth=0.5, zorder=3)
        for i in range(n_pts):
            ax1.annotate(f'U{i}', (td[i, 0], td[i, 1]),
                         textcoords='offset points', xytext=(6, 4),
                         color=text_color, fontsize=8)
        plt.colorbar(sc, ax=ax1, label='User Index',
                     shrink=0.8).ax.yaxis.set_tick_params(color=text_color)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('PCA — User Projection (PC1 vs PC2)')
        ax1.grid(True, alpha=0.2, color='#30363d')

    plt.tight_layout()
    return jsonify({'image': fig_to_base64(fig)})


@app.route('/api/chart/stats')
def chart_stats():
    """Dataset statistics chart."""
    if state['ratings_matrix'] is None:
        return jsonify({'error': 'No data loaded'}), 400

    rm = state['ratings_matrix']
    text_color = '#c9d1d9'
    fig, axes = dark_fig(2, 2, figsize=(12, 8))
    ax = axes.flatten()

    # Rating distribution
    ratings_flat = rm[rm > 0]
    ax[0].hist(ratings_flat, bins=20, color='#6c63ff', edgecolor='#0d1117', alpha=0.85)
    ax[0].set_title('Rating Distribution')
    ax[0].set_xlabel('Rating')
    ax[0].set_ylabel('Frequency')
    ax[0].grid(True, alpha=0.2, color='#30363d', axis='y')

    # Ratings per user
    per_user = [int(np.count_nonzero(rm[i])) for i in range(rm.shape[0])]
    ax[1].bar(range(len(per_user)), per_user, color='#2dd4bf', alpha=0.85)
    ax[1].set_title('Ratings per User')
    ax[1].set_xlabel('User Index')
    ax[1].set_ylabel('# Ratings')
    ax[1].grid(True, alpha=0.2, color='#30363d', axis='y')

    # Department / first categorical column distribution
    cat_col = next(
        (c for c in state['item_df'].columns
         if c != 'item_id' and (
             state['item_df'][c].dtype == object or
             state['item_df'][c].nunique() < 30
         )),
        None
    )
    if cat_col:
        dept = state['item_df'][cat_col].value_counts().head(10)
        ax[2].barh(dept.index.astype(str), dept.values, color='#f97316', alpha=0.85)
        ax[2].set_title(f'Items by {cat_col.capitalize()}')
        ax[2].set_xlabel('# Items')
        ax[2].grid(True, alpha=0.2, color='#30363d', axis='x')
    else:
        ax[2].text(0.5, 0.5, 'No categorical column found', ha='center', va='center',
                   color=text_color, transform=ax[2].transAxes)
        ax[2].set_title('Category Distribution')

    # First numeric column distribution
    num_col = next(
        (c for c in state['item_df'].columns
         if c != 'item_id' and
         pd.api.types.is_numeric_dtype(state['item_df'][c]) and
         state['item_df'][c].nunique() >= 3),
        None
    )
    if num_col:
        ax[3].hist(state['item_df'][num_col].dropna(), bins=15,
                   color='#ec4899', edgecolor='#0d1117', alpha=0.85)
        ax[3].set_title(f'Item {num_col.capitalize()} Distribution')
        ax[3].set_xlabel(num_col.capitalize())
        ax[3].set_ylabel('Frequency')
        ax[3].grid(True, alpha=0.2, color='#30363d', axis='y')
    else:
        ax[3].text(0.5, 0.5, 'No numeric column found', ha='center', va='center',
                   color=text_color, transform=ax[3].transAxes)
        ax[3].set_title('Numeric Distribution')

    for a in ax:
        a.tick_params(colors=text_color)

    plt.tight_layout()
    return jsonify({'image': fig_to_base64(fig)})


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route('/api/users')
def api_users():
    if state['user_df'] is None:
        return jsonify([])
    return jsonify(state['user_df']['user_id'].tolist())


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """Return recommendations as JSON."""
    body = request.get_json(force=True)
    user_id = body.get('user_id')
    method = body.get('method', 'svd').lower()
    n = int(body.get('n', 10))

    if state['miner'] is None:
        return jsonify({'error': 'Data not loaded'}), 400

    uid_dtype = state['user_df']['user_id'].dtype
    try:
        if np.issubdtype(uid_dtype, np.integer):
            search_id = int(user_id)
        elif np.issubdtype(uid_dtype, np.floating):
            search_id = float(user_id)
        else:
            search_id = str(user_id)
    except (ValueError, TypeError):
        search_id = user_id

    matches = np.where(state['user_df']['user_id'].values == search_id)[0]
    if len(matches) == 0:
        return jsonify({'error': f"User '{user_id}' not found"}), 404
    user_idx = int(matches[0])

    try:
        if method == 'svd':
            if state['svd_results'] is None:
                return jsonify({'error': 'Run SVD analysis first'}), 400
            recs = state['miner'].recommend_items_svd(user_idx, n)
        else:
            if state['nmf_results'] is None:
                return jsonify({'error': 'Run NMF analysis first'}), 400
            recs = state['miner'].recommend_items_nmf(user_idx, n)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Convert to JSON-safe list
    records = []
    for _, row in recs.iterrows():
        rec = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = round(float(v), 4)
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                rec[k] = None
            else:
                rec[k] = v
        records.append(rec)

    # User info
    user_info = state['user_df'].iloc[user_idx].to_dict()
    user_info_safe = {}
    for k, v in user_info.items():
        if isinstance(v, (np.integer,)):
            user_info_safe[k] = int(v)
        elif isinstance(v, (np.floating,)):
            user_info_safe[k] = round(float(v), 4)
        else:
            user_info_safe[k] = str(v)

    # Current ratings
    rated_indices = np.where(state['ratings_matrix'][user_idx] > 0)[0]
    current_ratings = []
    for idx in rated_indices[:15]:
        item = state['item_df'].iloc[idx]
        current_ratings.append({
            'item_id': str(item['item_id']),
            'rating': round(float(state['ratings_matrix'][user_idx, idx]), 1),
        })

    return jsonify({
        'user_info': user_info_safe,
        'recommendations': records,
        'current_ratings': current_ratings,
        'total_rated': int(len(rated_indices)),
    })


@app.route('/api/reload', methods=['POST'])
def api_reload():
    """Reload / regenerate data."""
    mode = request.get_json(force=True).get('mode', 'auto')
    if mode == 'regenerate':
        generator = AcademicDataGenerator(n_users=15, n_items=10000, sparsity=0.95)
        state['item_df'], state['user_df'], state['ratings_matrix'] = generator.generate_dataset()
        state['miner'] = MatrixMiner(state['ratings_matrix'], state['item_df'], state['user_df'])
        state['svd_results'] = state['nmf_results'] = state['pca_results'] = None
        state['data_source'] = 'Synthetic Data (Regenerated)'
    else:
        state['svd_results'] = state['nmf_results'] = state['pca_results'] = None
        load_application_data()
    return jsonify({'ok': True, 'data_source': state['data_source']})


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------
def _serialize_results(key):
    """Return a JSON-safe dict of analysis results, or None."""
    res = state.get(f'{key}_results')
    if res is None:
        return None
    safe = {}
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            safe[k] = int(v)
        elif isinstance(v, (np.floating,)):
            safe[k] = float(v)
        else:
            safe[k] = v
    return safe


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Working directory should be the MineMatrix folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_application_data()
    print('\n' + '='*60)
    print('  MineMatrix Flask App')
    print('  Open http://127.0.0.1:5000 in your browser')
    print('='*60 + '\n')
    app.run(debug=True, threaded=True, use_reloader=False)
