import streamlit as st
import pandas as pd
import requests
import re
from main import (
    get_recommendations_for_user,
    SVDRecommender,
    HybridRecommender,
    UserBasedCF,
    ItemBasedCF
)

# Page Configuration
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources from main.py
import main
main.load_resources()

# Extract valid user range
if main.train_user_item_matrix is not None:
    valid_users = main.train_user_item_matrix.index
    min_user_id = int(valid_users.min())
    max_user_id = int(valid_users.max())
    default_user_id = min_user_id
    num_users = len(valid_users)
else:
    min_user_id = 1
    max_user_id = 1000000
    default_user_id = 25
    num_users = 0

# Helper function to clean MovieLens title for TMDb search
def clean_title(title):
    # 1. Extract year e.g. Toy Story (1995) -> ('Toy Story', '1995')
    match = re.search(r'(.*?)\s*\((\d{4})\)\s*$', title)
    if match:
        query = match.group(1).strip()
        year = match.group(2)
    else:
        query = title.strip()
        year = None
        
    # 2. Remove alternative titles in parentheses e.g. (Postino, Il) or (a.k.a. Se7en)
    query = re.sub(r'\s*\((a\.k\.a\..*?|[^)]*,[^)]*)\)', '', query).strip()
    
    # 3. Rearrange trailing articles (e.g. "Postman, The" -> "The Postman")
    articles = [', The', ', A', ', An']
    for art in articles:
        if query.endswith(art):
            query = art[2:] + ' ' + query[:-len(art)]
            break
            
    return query.strip(), year

# Cached function to fetch movie poster and overview from TMDb
@st.cache_data(show_spinner=False)
def fetch_movie_poster_and_overview(title):
    api_key = "b29cbb37e08c275af122cd904b52012e"
    query, year = clean_title(title)
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": query}
    if year:
        params["year"] = year
    
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get("results"):
                movie = data["results"][0]
                poster_path = movie.get("poster_path")
                overview = movie.get("overview", "No synopsis available.")
                if not overview:
                    overview = "No synopsis available for this movie."
                if poster_path:
                    return "https://image.tmdb.org/t/p/w500" + poster_path, overview
    except Exception:
        pass
        
    # Placeholder image and text
    placeholder_url = "https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&q=80&w=400"
    return placeholder_url, "No synopsis available for this movie."

# Helper to fetch user watch history
def get_user_history(user_id, n_history=5):
    try:
        if main.train_user_item_matrix is not None:
            user_ratings = main.train_user_item_matrix.loc[user_id].dropna()
            top_rated = user_ratings.sort_values(ascending=False).head(n_history)
            
            history_results = []
            for movie_id, rating in top_rated.items():
                movie_title = main.movies_df[main.movies_df['movieId'] == movie_id]['title'].values[0] if movie_id in main.movies_df['movieId'].values else f"Movie {movie_id}"
                genre = main.movies_df[main.movies_df['movieId'] == movie_id]['genres'].values[0] if movie_id in main.movies_df['movieId'].values else "Unknown"
                history_results.append({
                    'Movie': movie_title,
                    'Genre': genre,
                    'Rating': float(rating)
                })
            return history_results
    except Exception:
        pass
    return []

# Custom Styling (Glassmorphism, Netflix Red & Dark Mode Grid)
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 17, 22, 1) 0%, rgba(26, 31, 41, 1) 90.1%);
    }
    /* Header styling */
    .title-text {
        background: linear-gradient(90deg, #ff4b4b, #ff7e5f, #feb47b);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
        animation: gradient-shift 15s ease infinite;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Grid System */
    .netflix-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 25px;
        padding: 20px 0;
    }
    
    .netflix-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease, box-shadow 0.3s ease;
        display: flex;
        flex-direction: column;
        height: 100%;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    .netflix-card:hover {
        transform: translateY(-8px) scale(1.03);
        border-color: rgba(229, 9, 20, 0.85); /* Netflix Red */
        box-shadow: 0 12px 24px rgba(229, 9, 20, 0.25);
    }
    
    .netflix-poster-container {
        position: relative;
        width: 100%;
        padding-top: 150%; /* 2:3 Aspect Ratio */
        overflow: hidden;
        background: #181818;
    }
    
    .netflix-poster {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    
    .netflix-card:hover .netflix-poster {
        transform: scale(1.08);
    }
    
    .netflix-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(15, 17, 22, 0.93);
        opacity: 0;
        transition: opacity 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 18px;
        box-sizing: border-box;
        color: #f1f5f9;
        font-size: 0.8rem;
        text-align: left;
        overflow-y: auto;
    }
    
    .netflix-card:hover .netflix-overlay {
        opacity: 1;
    }
    
    .netflix-details {
        padding: 15px;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        justify-content: space-between;
        background: rgba(15, 17, 22, 0.85);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .netflix-movie-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.75rem;
    }
    
    .netflix-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 8px;
    }
    
    .netflix-badge-rating {
        background: linear-gradient(135deg, #e50914 0%, #b81d24 100%);
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 2px 4px rgba(229, 9, 20, 0.2);
    }
    
    .netflix-badge-genre {
        background: rgba(255, 255, 255, 0.08);
        color: #cccccc;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        max-width: 105px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Main Title Section
st.markdown('<div class="title-text">🎬 Premium Movie Recommendations</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8; font-size:1.15rem; margin-bottom:2rem;">State-of-the-art Collaborative Filtering and Matrix Factorization (SVD) Recommendation Engine.</p>', unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.markdown('<h2 style="color: #ff4b4b;">⚙️ Engine Controls</h2>', unsafe_allow_html=True)
if num_users > 0:
    st.sidebar.info(f"💡 **Dataset active**: {num_users} users loaded.\nRange: **{min_user_id}** to **{max_user_id}**")
st.sidebar.markdown('---')

user_id = st.sidebar.number_input(
    "👥 Select Target User ID",
    min_value=min_user_id,
    max_value=max_user_id,
    value=default_user_id,
    help=f"Enter a valid user ID between {min_user_id} and {max_user_id}."
)

method = st.sidebar.selectbox(
    "🧠 Recommendation Algorithm",
    ["user", "item", "svd", "hybrid"],
    format_func=lambda x: {
        "user": "👥 User-Based CF (Cosine Sim)", 
        "item": "🍿 Item-Based CF (Cosine Sim)", 
        "svd": "⚡ Matrix Factorization (SVD)",
        "hybrid": "💎 Hybrid Ensemble System"
    }[x],
    help="Choose the algorithm to power the recommendations."
)

n_recommendations = st.sidebar.slider(
    "📊 Recommendation Count",
    min_value=5,
    max_value=20,
    value=10,
    help="Number of recommendations to retrieve and plot."
)

st.sidebar.markdown('---')
st.sidebar.markdown('<h3 style="color: #feb47b;">🔍 Discovery Filters</h3>', unsafe_allow_html=True)

genres_list = ["All", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
               "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
               "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

selected_genre = st.sidebar.selectbox(
    "Filter Recommendations by Genre",
    genres_list,
    index=0,
    help="Restrict the generated list to a specific movie genre."
)

st.sidebar.markdown('---')
run_button = st.sidebar.button("🚀 Generate Recommendations", use_container_width=True)

# Main Content Layout
if run_button:
    # 1. Fetch User History Shelf
    history = get_user_history(user_id)
    if history:
        st.markdown('### 🍿 Your Watch History (Historically Top Rated)')
        history_html = '<div class="netflix-grid" style="grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 15px; padding: 10px 0;">'
        for item in history:
            poster_url, _ = fetch_movie_poster_and_overview(item['Movie'])
            safe_hist_title = item['Movie'].replace('"', '&quot;')
            history_html += f"""
<div class="netflix-card" style="opacity: 0.85;">
<div class="netflix-poster-container" style="padding-top: 150%;">
<img class="netflix-poster" src="{poster_url}" alt="{safe_hist_title}" />
</div>
<div class="netflix-details" style="padding: 10px; background: rgba(30, 41, 59, 0.65);">
<div class="netflix-movie-title" style="font-size: 0.9rem; height: 2.2rem; -webkit-line-clamp: 2;">{item['Movie']}</div>
<div class="netflix-meta" style="margin-top: 4px;">
<span class="netflix-badge-rating" style="background: #ffaa00; font-size: 0.75rem; padding: 2px 6px;">★ {item['Rating']:.1f}</span>
<span class="netflix-badge-genre" style="font-size: 0.7rem; padding: 2px 6px; max-width: 75px;">{item['Genre'].split('|')[0]}</span>
</div>
</div>
</div>
"""
        history_html += '</div>'
        clean_hist_html = "\n".join(line.strip() for line in history_html.split("\n"))
        st.markdown(clean_hist_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # 2. Fetch and filter Recommendations
    with st.spinner("🧠 Querying recommendation model and retrieving metadata..."):
        # Fetch 100 entries so we have enough candidates to apply genre filtering
        message, df = get_recommendations_for_user(user_id, method, n_recommendations=100)
        
    if not df.empty:
        # Apply Genre Filtering
        if selected_genre != "All":
            df = df[df['Genre'].str.contains(selected_genre, case=False, na=False)]
            message = f"{message} (Filtered by: {selected_genre})"
            
        # Squeeze down to the requested size
        df = df.head(n_recommendations).reset_index(drop=True)
        
        if not df.empty:
            st.markdown(f"### ✨ {message}")
            
            # Netflix-Style Movie Posters Grid
            grid_html = '<div class="netflix-grid">'
            for idx, row in df.iterrows():
                poster_url, overview = fetch_movie_poster_and_overview(row['Movie'])
                # Format predicted rating to float for truncation
                rating_val = float(row['Predicted Rating'])
                formatted_rating = f"{rating_val:.1f}"
                safe_title = row['Movie'].replace('"', '&quot;')
                
                grid_html += f"""
<div class="netflix-card">
<div class="netflix-poster-container">
<img class="netflix-poster" src="{poster_url}" alt="{safe_title}" />
<div class="netflix-overlay">
<div>
<div style="font-weight:800; font-size: 0.9rem; margin-bottom:8px; color:#ff4b4b; text-transform:uppercase; letter-spacing:1px;">Synopsis</div>
<div style="line-height:1.4;">{overview}</div>
</div>
</div>
</div>
<div class="netflix-details">
<div class="netflix-movie-title" title="{safe_title}">{row['Movie']}</div>
<div class="netflix-meta">
<span class="netflix-badge-rating">★ {formatted_rating}</span>
<span class="netflix-badge-genre" title="{row['Genre']}">{row['Genre']}</span>
</div>
</div>
</div>
"""
            grid_html += '</div>'
            grid_html_clean = "\n".join(line.strip() for line in grid_html.split("\n"))
            st.markdown(grid_html_clean, unsafe_allow_html=True)
            
            # Collapsible Analytical Metrics Expandable Section
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 Show Analytical Metrics & Data Visualization", expanded=False):
                col1, col2 = st.columns([1.2, 1])
                
                with col1:
                    st.markdown('<p style="color: #ff7e5f; font-weight: 600; font-size: 1.1rem; margin-top: 1rem;">Predicted Rating Visual Comparison</p>', unsafe_allow_html=True)
                    
                    # Format predicted rating to float for plotting
                    plot_df = df.copy()
                    plot_df['Predicted Rating'] = plot_df['Predicted Rating'].astype(float)
                    
                    # Display interactive Streamlit bar chart
                    st.bar_chart(
                        plot_df.set_index('Movie')['Predicted Rating'],
                        color='#ff7e5f',
                        use_container_width=True
                    )
                    
                with col2:
                    st.markdown('<p style="color: #feb47b; font-weight: 600; font-size: 1.1rem; margin-top: 1rem;">Raw Recommendations Metrics Table</p>', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True)
        else:
            st.warning(f"⚠️ No recommendations found matching the genre **{selected_genre}**. Try adjusting your algorithm or count controls.")
            
    else:
        st.error(f"❌ {message}")
else:
    # Initial state screen
    st.info("👈 Configure engine parameters in the sidebar and click **Generate Recommendations** to begin!")
