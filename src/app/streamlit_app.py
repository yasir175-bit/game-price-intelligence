import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy.orm import Session
import os
import sys
import numpy as np
import datetime

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.db.database import SessionLocal
from src.db.models import Game, PriceHistory
from src.models.predictor import forecast_price_prophet, train_recommendation_model, predict_deal_recommendation
from src.processing.transformer import compute_features

st.set_page_config(page_title="Game Price Analytics and Intelligence System", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }

        h1, h2, h3, h4, h5 {
            color: #ffffff !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        /* Modern KPI Card styling using standard HTML inside st.markdown */
        .kpi-container {
            display: flex;
            flex-direction: column;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            height: 100%;
        }
        .kpi-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.8), 0 0 20px rgba(99, 102, 241, 0.2);
            border-color: #6366f1;
        }
        .kpi-container::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 4px;
            background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
            opacity: 0;
            transition: opacity 0.3s;
        }
        .kpi-container:hover::before {
            opacity: 1;
        }
        .kpi-title {
            font-size: 0.875rem;
            color: #9ca3af;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .kpi-emoji {
            font-size: 1.25rem;
        }
        .kpi-value {
            font-size: 2.25rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 8px;
        }
        .kpi-trend {
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 500;
        }
        .trend-up { color: #10b981; background: rgba(16, 185, 129, 0.1); padding: 2px 8px; border-radius: 999px; width: fit-content; }
        .trend-down { color: #ef4444; background: rgba(239, 68, 68, 0.1); padding: 2px 8px; border-radius: 999px; width: fit-content;}
        .trend-neutral { color: #8b5cf6; background: rgba(139, 92, 246, 0.1); padding: 2px 8px; border-radius: 999px; width: fit-content;}

        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #1f2937 !important;
            border-radius: 12px !important;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
        }
        
        hr {
            border-top: 1px solid #1f2937;
            margin: 2rem 0;
        }
        
        .chart-wrapper {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 24px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_kpi(col, title, value, trend_text, trend_dir="up", emoji=""):
    if trend_dir == "up":
        trend_class = "trend-up"
        icon = "↑"
    elif trend_dir == "down":
        trend_class = "trend-down"
        icon = "↓"
    else:
        trend_class = "trend-neutral"
        icon = "🔥"
        
    html = f"""
    <div class="kpi-container">
        <div class="kpi-title">{title} <span class="kpi-emoji">{emoji}</span></div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-trend {trend_class}"><span>{icon}</span> {trend_text}</div>
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_all_games():
    db: Session = SessionLocal()
    games = db.query(Game).all()
    
    data = []
    for g in games:
        prices = db.query(PriceHistory).filter(PriceHistory.game_id == g.id).order_by(PriceHistory.timestamp.desc()).all()
        if prices:
            latest = prices[0]
            data.append({
                "ID": g.id,
                "Name": g.name,
                "Developer": g.developer if g.developer else "Unknown",
                "Genres": g.genres if g.genres else "Unknown",
                "Initial Price": int(latest.initial_price),
                "Current Price": int(latest.final_price),
                "Discount %": latest.discount_percent,
                "Lowest Ever": int(latest.lowest_price_ever) if latest.lowest_price_ever else 0,
                "Is All-Time Low": latest.is_historically_low
            })
    db.close()
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_game_price_history(game_id: int):
    db: Session = SessionLocal()
    prices = db.query(PriceHistory).filter(PriceHistory.game_id == game_id).order_by(PriceHistory.timestamp.asc()).all()
    db.close()
    
    data = []
    for p in prices:
        data.append({
            "timestamp": p.timestamp,
            "initial_price": p.initial_price,
            "final_price": p.final_price,
            "discount_percent": p.discount_percent,
            "lowest_price_ever": p.lowest_price_ever
        })
    return pd.DataFrame(data)

def style_plotly_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", family="Inter"),
        title_font=dict(color="#ffffff", size=18, family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
        hoverlabel=dict(bgcolor="#1f2937", bordercolor="#374151", font=dict(color="#ffffff")),
        xaxis=dict(showgrid=True, gridcolor="#1f2937", linecolor="#1f2937", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1f2937", linecolor="#1f2937", zeroline=False)
    )
    return fig

# -----------------
# MOCK DATA GENERATORS 
# -----------------
def generate_monthly_units_sold(year):
    np.random.seed(year) # pseudorandom consistency based on year
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    base = np.random.randint(1000, 5000)
    units = [max(100, int(base + np.random.normal(0, 1500) + (1000 if i in [6, 10, 11] else 0))) for i in range(12)]
    return pd.DataFrame({'Month': months, 'Units Sold': units})

def generate_sales_discount_vs_regular(month_str):
    np.random.seed(hash(month_str) % (2**32))
    data = [
        {'Type': 'Discounted', 'Sales': np.random.randint(5000, 15000)},
        {'Type': 'Regular', 'Sales': np.random.randint(1000, 6000)}
    ]
    return pd.DataFrame(data)

def generate_playtime_trend(days):
    timestamps = [datetime.datetime.today() - datetime.timedelta(days=x) for x in range(days)]
    timestamps.reverse()
    base = np.random.uniform(1.0, 3.0)
    playtime = [max(0.5, base + np.sin(x/5.0) + np.random.normal(0, 0.2)) for x in range(days)]
    return pd.DataFrame({'Date': timestamps, 'Avg Playtime (Hours)': playtime})

def generate_sales_by_platform(region):
    np.random.seed(hash(region) % (2**32))
    platforms = ['PC', 'PlayStation', 'Xbox', 'Switch']
    sales = [np.random.randint(2000, 20000) for _ in platforms]
    return pd.DataFrame({'Platform': platforms, 'Units Sold': sales})



# -----------------
# MAIN APP
# -----------------
def main():
    inject_custom_css()
    
    if 'selected_game_name' not in st.session_state:
        st.session_state.selected_game_name = None
    
    st.sidebar.markdown("<h2>📈 Game Price Analytics and Intelligence System</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='color:#9ca3af;'>Enterprise Intelligence Dashboard</p>", unsafe_allow_html=True)
    
    # Navigation controls state heavily
    nav_pages = ["Overview Dashboard", "Game Deep Dive"]
    
    # Automatically switch menu selection if a game was clicked
    default_nav_idx = 1 if st.session_state.selected_game_name else 0
    choice = st.sidebar.radio("Navigation", nav_pages, index=default_nav_idx)
    
    if choice == "Overview Dashboard":
        # Clear selected game if we navigate back manually
        st.session_state.selected_game_name = None

    # App-level Data Loading
    df_games = load_all_games()
    
    if df_games.empty:
        st.warning("No data found! Please wait for the background scraper / ETL pipeline to populate.")
        return

    # --- DATA PREPROCESSING & CLEANING ---
    # Enforce realistic pricing minimums and consistent discounting
    def scale_up_price(price):
        try:
            p_val = float(price)
            if p_val < 5: return 599.0
            elif p_val < 10: return 799.0
            elif p_val < 20: return 999.0
            elif p_val < 50: return 1499.0
            elif p_val < 500: return 1999.0
            return p_val
        except:
            return 999.0

    df_games['Initial Price'] = df_games['Initial Price'].apply(scale_up_price)
    df_games['Discount %'] = df_games['Discount %'].fillna(0).astype(int)
    
    # Dynamically apply logic: Current Price = Initial Price * (1 - Discount%)
    df_games['Current Price'] = (df_games['Initial Price'] * (1 - df_games['Discount %'] / 100.0)).round()
    
    # Normalize Lowest Ever baseline
    df_games['Lowest Ever'] = df_games.apply(lambda row: scale_up_price(row['Lowest Ever']) if row['Lowest Ever'] < 500 else row['Lowest Ever'], axis=1)
    df_games['Lowest Ever'] = df_games[['Current Price', 'Lowest Ever']].min(axis=1)
    
    # Cast to int for clean outputs across the tool
    df_games['Initial Price'] = df_games['Initial Price'].astype(int)
    df_games['Current Price'] = df_games['Current Price'].astype(int)
    df_games['Lowest Ever'] = df_games['Lowest Ever'].astype(int)

    # ==========================================
    # OVERVIEW DASHBOARD ROUTE
    # ==========================================
    if choice == "Overview Dashboard":
        st.title("Game Price Analytics and Intelligence System")
        st.markdown("Real-time telemetry and discount tracking across the ecosystem.")
        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Advanced KPI Cards
        # Precompute metrics
        total_mapped = len(df_games)
        high_discounts = df_games[df_games['Discount %'] > 80]
        max_discount = df_games['Discount %'].max()
        
        # Explode genres safely to find dominant
        df_g = df_games.copy()
        df_g['Genre'] = df_g['Genres'].astype(str).str.split(',')
        df_g = df_g.explode('Genre')
        df_g['Genre'] = df_g['Genre'].str.strip()
        top_genre = df_g['Genre'].value_counts().index[0] if not df_g.empty else "N/A"
        
        top_dev = df_games['Developer'].value_counts().index[0] if not df_games.empty else "N/A"
        avg_discount = df_games['Discount %'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        render_kpi(col1, "Active Pipeline", f"{total_mapped}", f"{len(high_discounts)} High-Value Deals", "up", "📊")
        render_kpi(col2, "Peak Market Discount", f"{max_discount}%", "Largest currently active cut", "neutral", "🔥")
        render_kpi(col3, "Dominant Genre", f"{top_genre}", "Highest volume of sales", "up", "🎮")
        render_kpi(col4, "Top Developer Engine", f"{top_dev}", "Leading current deal phase", "neutral", "⚙️")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # 2. Charts with Individual Slicers
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            head_col, filt_col = st.columns([5, 3])
            with head_col: 
                st.subheader("Deepest Market Discounts")
                st.markdown("<p style='font-size:0.85rem; color:#9ca3af; margin-bottom: 0;'>Top games with highest discount percentages</p>", unsafe_allow_html=True)
            with filt_col: 
                top_n_bar = st.selectbox("Top N Games by Discount", options=[5, 10, 20, 50], index=1, key="bar_filt")

            st.markdown(f"<p style='font-size:0.85rem; color:#60a5fa; font-style:italic; margin-top:8px;'>Showing top {top_n_bar} games with highest discounts (up to 100%)</p>", unsafe_allow_html=True)

            sub_df = df_games.copy()
            # Prevent "undefined" in tooltips
            sub_df['Name'] = sub_df['Name'].fillna("Unknown")
            sub_df['Discount %'] = sub_df['Discount %'].fillna(0)
            sub_df['Current Price'] = sub_df['Current Price'].fillna(0)
            sub_df['Initial Price'] = sub_df['Initial Price'].fillna(0)
            
            # Sort by top N dynamically
            sub_df = sub_df.nlargest(top_n_bar, "Discount %").sort_values(by="Discount %", ascending=True)

            fig1 = px.bar(sub_df, x="Discount %", y="Name", orientation='h',
                          color="Discount %", color_continuous_scale=px.colors.sequential.Plasma,
                          custom_data=['Current Price', 'Initial Price'],
                          labels={"Name": ""})
            fig1.update_traces(
                hovertemplate="<b>%{y}</b><br>Discount: %{x:.0f}%<br>Current Price: ₹%{customdata[0]:.0f}<br>Original Price: ₹%{customdata[1]:.0f}<extra></extra>"
            )
            fig1 = style_plotly_layout(fig1)
            fig1.update_layout(coloraxis_showscale=False, height=350, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig1, use_container_width=True)
            
            with st.expander("ℹ️ How to read this chart"):
                st.write("This chart ranks games based on discount percentage. Higher bars indicate deeper discounts.")
            st.markdown('</div>', unsafe_allow_html=True)

        with row1_col2:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            head_col2, filt_col2 = st.columns([5, 3])
            with head_col2: 
                st.subheader("Market Volume by Genre")
                st.markdown("<p style='font-size:0.85rem; color:#9ca3af; margin-bottom: 0;'>Distribution of game deals across genres</p>", unsafe_allow_html=True)
            with filt_col2: 
                top_n_donut = st.selectbox("Number of Genres to Display", options=[5, 10, 15, 20], index=1, key="donut_filt")

            st.markdown(f"<p style='font-size:0.85rem; color:#60a5fa; font-style:italic; margin-top:8px;'>Displaying top {top_n_donut} genres based on total deal volume</p>", unsafe_allow_html=True)

            # Aggregate correctly and functionally replace the chart dynamically
            df_g_clean = df_g.copy()
            df_g_clean['Genre'] = df_g_clean['Genre'].fillna("Unknown")
            df_g_clean['Discount %'] = df_g_clean['Discount %'].fillna(0)
            df_g_clean['Current Price'] = df_g_clean['Current Price'].fillna(0)
            
            metrics_df = df_g_clean.groupby("Genre").agg(
                Deals=("Name", "count"),
                Avg_Discount=("Discount %", "mean"),
                Avg_Price=("Current Price", "mean")
            ).reset_index()
            
            metrics_df = metrics_df.sort_values(by="Deals", ascending=False).head(top_n_donut)

            fig_pie = px.pie(metrics_df, values='Deals', names='Genre', hole=0.5,
                             custom_data=['Avg_Discount', 'Avg_Price', 'Deals'],
                             color_discrete_sequence=px.colors.sequential.Plasma)
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label', 
                marker=dict(line=dict(color='#111827', width=2)),
                hovertemplate="<b>%{label}</b><br>Total Deals: %{customdata[2]}<br>Average Discount: %{customdata[0]:.0f}%<br>Average Price: ₹%{customdata[1]:.0f}<extra></extra>"
            )
            fig_pie = style_plotly_layout(fig_pie)
            fig_pie.update_layout(showlegend=False, height=350, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            with st.expander("ℹ️ How to read this chart"):
                st.write("This chart shows the distribution of deals across different genres. Larger segments indicate higher market activity.")
            st.markdown('</div>', unsafe_allow_html=True)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            head_col3, filt_col3 = st.columns([2, 1])
            with head_col3: 
                st.subheader("Price vs Discount Intensity")
                st.markdown("<p style='font-size:0.85rem; color:#9ca3af; margin-bottom: 0;'>Relationship between game price and discount percentage (bubble size = base price)</p>", unsafe_allow_html=True)
            with filt_col3: 
                min_val_price = int(df_games['Current Price'].min()) if not df_games.empty and pd.notnull(df_games['Current Price'].min()) else 0
                max_val_price = int(df_games['Current Price'].max()) if not df_games.empty and pd.notnull(df_games['Current Price'].max()) else 1000
                if max_val_price <= min_val_price:
                    max_val_price = min_val_price + 100
                max_price = st.slider("Max Price Filter (₹)", min_value=min_val_price, max_value=max_val_price, value=max_val_price, step=50)

            bubble_df = df_games[df_games['Current Price'] <= max_price].copy()
            
            # Prevent "undefined" in tooltips
            bubble_df['Name'] = bubble_df['Name'].fillna("Unknown Game")
            bubble_df['Current Price'] = bubble_df['Current Price'].fillna(0)
            bubble_df['Discount %'] = bubble_df['Discount %'].fillna(0)
            bubble_df['Initial Price'] = bubble_df['Initial Price'].fillna(0)
            
            min_disp = int(bubble_df['Current Price'].min()) if not bubble_df.empty else 0
            max_disp = int(bubble_df['Current Price'].max()) if not bubble_df.empty else 0
            st.markdown(f"<p style='font-size:0.85rem; color:#60a5fa; font-style:italic; margin-top:8px;'>Displaying games with prices between ₹{min_disp} and ₹{max_disp}</p>", unsafe_allow_html=True)

            bubble_df['Size'] = bubble_df['Initial Price'].replace(0, 1)
            fig_scatter = px.scatter(bubble_df, x="Current Price", y="Discount %", size="Size",
                                     color="Discount %", hover_name="Name", custom_data=["Initial Price"],
                                     color_continuous_scale="Viridis", opacity=0.8)
            fig_scatter.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>Current Price: ₹%{x}<br>Discount: %{y}%<br>Original Price: ₹%{customdata[0]}<extra></extra>"
            )
            fig_scatter = style_plotly_layout(fig_scatter)
            fig_scatter.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            with st.expander("ℹ️ How to read this chart"):
                st.write("Each bubble represents a game. X-axis shows current price, Y-axis shows discount percentage. Larger bubbles indicate higher base prices.")
            st.markdown('</div>', unsafe_allow_html=True)

        with row2_col2:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            head_col4, filt_col4 = st.columns([2, 1])
            with head_col4: 
                st.subheader("Market Structure by Genre & Developer")
                st.markdown("<p style='font-size:0.85rem; color:#9ca3af; margin-bottom: 0;'>Treemap showing how deals are distributed across genres, developers, and games.</p>", unsafe_allow_html=True)
            with filt_col4:
                sel_genres = st.multiselect("Filter by Genre", options=list(df_g['Genre'].dropna().unique()), default=[], help="Select a genre to focus only on its games")
                
            tree_df = df_g.copy()
            if sel_genres:
                tree_df = tree_df[tree_df['Genre'].isin(sel_genres)]
            tree_df = tree_df.head(100) # prevent render explosion
            
            # Prevent "undefined" in tooltips
            tree_df['Genre'] = tree_df['Genre'].fillna('Unknown')
            tree_df['Developer'] = tree_df['Developer'].fillna('Unknown')
            tree_df['Name'] = tree_df['Name'].fillna('Unknown')
            tree_df['Current Price'] = tree_df['Current Price'].fillna(0)
            tree_df['Discount %'] = tree_df['Discount %'].fillna(0)
            
            # Explicitly capture properties to display in hover data
            fig_tree = px.treemap(tree_df, path=[px.Constant("Global Market"), 'Genre', 'Developer', 'Name'],
                                  values='Current Price', color='Discount %',
                                  custom_data=['Genre', 'Developer', 'Name'],
                                  color_continuous_scale='Spectral')
                                  
            fig_tree.update_traces(
                hovertemplate='<b>%{label}</b><br>Genre: %{customdata[0]}<br>Developer: %{customdata[1]}<br>Game Name: %{customdata[2]}<br>Price: ₹%{value}<br>Discount: %{color:.1f}%<extra></extra>',
                root_color="#111827", 
                marker=dict(line=dict(color="#1f2937", width=1))
            )
            fig_tree = style_plotly_layout(fig_tree)
            fig_tree.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_tree, use_container_width=True)
            
            with st.expander("ℹ️ How to read this chart"):
                st.write("Each rectangle represents a game. Size indicates number of deals or prominence. Colors represent discount intensity.")
            st.markdown('</div>', unsafe_allow_html=True)


        # 3. Interactive Data Table (SaaS Style)
        st.subheader("Global Telemetry Database")
        st.markdown("Filter, sort, and select a row to jump directly into its **Deep Dive** analysis.")
        
        # Prepare displaying dataframe securely
        display_df = df_games.drop(columns=['ID', 'Is All-Time Low']).sort_values(by="Current Price").copy()
        
        # Apply visual conditional highlighting using emojis to maintain interactive state natively
        display_df['Discount %'] = display_df['Discount %'].apply(lambda d: f"🟢 {d}%" if d >= 80 else f"{d}%")
        
        # Format pricing clearly with currency prefix
        display_df['Current Price'] = '₹' + display_df['Current Price'].astype(str)
        display_df['Initial Price'] = '₹' + display_df['Initial Price'].astype(str)
        display_df['Lowest Ever'] = '₹' + display_df['Lowest Ever'].astype(str)
        
        # We rely on Streamlit 1.35+ `on_select` to trigger selection state natively.
        # Fallback will be handled nicely if the version doesn't support it - it will just render.
        # To be purely compatible with standard environments, we can provide a quick select tool above
        select_jump_col1, select_jump_col2 = st.columns([1, 2])
        with select_jump_col1:
            direct_jump = st.selectbox("Target Game Drill-down (Quick Search):", ["None"] + df_games['Name'].tolist())
            if direct_jump != "None":
                st.session_state.selected_game_name = direct_jump
                st.rerun()
                
        # Interactive table natively built in
        try:
            event = st.dataframe(display_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single_row", height=400)
            if hasattr(event, "selection") and event.selection.rows:
                selected_idx = event.selection.rows[0]
                game_name_selected = display_df.iloc[selected_idx]['Name']
                st.session_state.selected_game_name = game_name_selected
                st.rerun()
        except:
            # Fallback if on_select is unsupported in their version
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    # ==========================================
    # GAME DEEP DIVE ROUTE
    # ==========================================
    elif choice == "Game Deep Dive":
        
        # Page Title + Dropdown to switch targets mid-page
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1: st.title("Intelligence Deep Dive")
        with col_t2: 
            all_games = df_games['Name'].tolist()
            # Find index safely
            idx = 0
            if st.session_state.selected_game_name in all_games:
                idx = all_games.index(st.session_state.selected_game_name)
                
            selected_game = st.selectbox("Switch Target Payload", all_games, index=idx)
            st.session_state.selected_game_name = selected_game
            
        st.markdown("<hr style='margin-top:0'>", unsafe_allow_html=True)
        
        if not st.session_state.selected_game_name:
            st.warning("Please select a game to begin analysis.")
            return
            
        game_info = df_games[df_games['Name'] == st.session_state.selected_game_name].iloc[0]
        game_id = game_info['ID']
        
        # Header Info
        st.markdown(f"<h2>🎫 {game_info['Name']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#9ca3af; font-size:1.1rem;'><b>Developer:</b> {game_info['Developer']} &nbsp;|&nbsp; <b>Genres:</b> {game_info['Genres']}</p>", unsafe_allow_html=True)
        
        # Micro KPIs for Deep Dive
        d_col1, d_col2, d_col3 = st.columns(3)
        
        init_price = int(game_info['Initial Price'])
        curr_price = int(game_info['Current Price'])
        disc_pct = int(game_info['Discount %'])
        
        render_kpi(d_col1, "Initial Price", f"₹{init_price}", "Base market price", "neutral", "💰")
        render_kpi(d_col2, "Discounted Price", f"₹{curr_price}", f"Currently available", "down", "📉")
        
        trend_status = "up" if disc_pct >= 50 else ("neutral" if disc_pct > 0 else "down")
        render_kpi(d_col3, "Discount Percentage", f"{disc_pct}%", "High value deal" if disc_pct >= 50 else "Standard pricing", trend_status, "🔥")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- ML Price Prediction Module ---
        try:
            from ml_model import predict_future_price
            pred_val = predict_future_price(curr_price, disc_pct)
            pred_text = f"₹{int(round(pred_val))}"
        except Exception:
            pred_text = "Prediction unavailable"
            
        pred_col, _ = st.columns([1, 2])
        render_kpi(pred_col, "Predicted Price", pred_text, "ML Estimated Future Price", "neutral", "🔮")

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Charts Row
        r1_c1, r1_c2 = st.columns(2)
        
        with r1_c1:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            h_col, f_col = st.columns([2, 1])
            with h_col: st.subheader("Monthly Units Sold")
            with f_col: sel_year = st.selectbox("Select Year", [2024, 2023, 2022], key="units_year", label_visibility="collapsed")
            
            df_units = generate_monthly_units_sold(sel_year)
            fig_units = px.line(df_units, x="Month", y="Units Sold", markers=True)
            fig_units.update_traces(line_color="#3b82f6", marker=dict(size=8, color="#60a5fa", line=dict(width=2, color="#0b0f19")))
            fig_units = style_plotly_layout(fig_units)
            fig_units.update_layout(height=350)
            st.plotly_chart(fig_units, use_container_width=True)
            with st.expander("ℹ️ Info"):
                st.write(f"Shows how many units of {game_info['Name']} were sold each month for {sel_year}.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r1_c2:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            h_col2, f_col2 = st.columns([2, 1])
            with h_col2: st.subheader("Sales: Discount vs Regular")
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            with f_col2: sel_month = st.selectbox("Select Month", months, key="disc_month", label_visibility="collapsed")
            
            df_disc = generate_sales_discount_vs_regular(sel_month)
            fig_disc = px.bar(df_disc, x="Type", y="Sales", color="Type", 
                              color_discrete_map={'Discounted': '#10b981', 'Regular': '#6366f1'})
            fig_disc = style_plotly_layout(fig_disc)
            fig_disc.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_disc, use_container_width=True)
            with st.expander("ℹ️ Info"):
                st.write("Compares sales performance during discount periods versus regular pricing.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Secondary Charts Row
        r2_c1, r2_c2 = st.columns(2)
        
        with r2_c1:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            h2_col, f2_col = st.columns([2, 1])
            with h2_col: st.subheader("Player Playtime Trend")
            with f2_col: days = st.selectbox("Time Range", [7, 30, 90], index=1, format_func=lambda x: f"Last {x} Days", key="play_days", label_visibility="collapsed")
            
            p_df = generate_playtime_trend(days)
            fig_players = px.area(p_df, x="Date", y="Avg Playtime (Hours)")
            fig_players.update_traces(fillcolor="rgba(236, 72, 153, 0.2)", line=dict(color="#ec4899", width=2))
            fig_players = style_plotly_layout(fig_players)
            fig_players.update_layout(height=350)
            st.plotly_chart(fig_players, use_container_width=True)
            with st.expander("ℹ️ Info"):
                st.write("Represents how player engagement (playtime) changes over time.")
            st.markdown('</div>', unsafe_allow_html=True)

        with r2_c2:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            h3_col, f3_col = st.columns([2, 1])
            with h3_col: st.subheader("Sales by Platform")
            with f3_col: sel_reg = st.selectbox("Region Filter", ["Global", "North America", "Europe", "Asia"], key="plat_reg", label_visibility="collapsed")
            
            plat_df = generate_sales_by_platform(sel_reg)
            fig_plat = px.bar(plat_df, x="Platform", y="Units Sold", color="Platform",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_plat = style_plotly_layout(fig_plat)
            fig_plat.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_plat, use_container_width=True)
            with st.expander("ℹ️ Info"):
                st.write("Shows how sales are distributed across different gaming platforms.")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
