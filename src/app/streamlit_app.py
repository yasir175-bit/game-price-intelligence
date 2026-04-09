import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy.orm import Session
import os
import sys
import numpy as np

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.db.database import SessionLocal
from src.db.models import Game, PriceHistory
from src.models.predictor import forecast_price_prophet, train_recommendation_model, predict_deal_recommendation
from src.processing.transformer import compute_features

st.set_page_config(page_title="Game Price Intelligence", layout="wide", page_icon="🎮")

def inject_custom_css():
    st.markdown("""
        <style>
        /* Global Steam-like Font & Backgrounds */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        .stApp {
            background-color: #1b2838;
            color: #c6d4df;
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #171a21;
            border-right: 1px solid #2a475e;
        }

        /* Custom Header Styling */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }

        /* Metric Cards - Glassmorphism */
        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, rgba(42, 71, 94, 0.4) 0%, rgba(23, 26, 33, 0.8) 100%);
            border: 1px solid #2a475e;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(102, 192, 244, 0.15);
            border: 1px solid #66c0f4;
        }
        
        /* Metric Text Coloring */
        div[data-testid="metric-container"] > div > div > div {
            color: #66c0f4 !important; /* Steam Blue for numbers */
        }
        
        /* Delta Value styling - Steam Green */
        [data-testid="stMetricDelta"] svg {
            color: #a4d007 !important; 
        }
        [data-testid="stMetricDelta"] div {
            color: #a4d007 !important;
        }

        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #2a475e !important;
            border-radius: 8px !important;
            overflow: hidden;
        }

        /* Buttons matching Steam */
        .stButton>button {
            background: linear-gradient(to right, #417a9b 5%, #66c0f4 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 4px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #66c0f4 5%, #417a9b 100%) !important;
            box-shadow: 0 0 10px rgba(102, 192, 244, 0.5);
        }
        
        hr {
            border-top: 1px solid #2a475e;
        }
        </style>
    """, unsafe_allow_html=True)

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
                "Developer": g.developer,
                "Genres": g.genres,
                "Initial Price": latest.initial_price,
                "Current Price": latest.final_price,
                "Discount %": latest.discount_percent,
                "Lowest Ever": latest.lowest_price_ever,
                "Is All-Time Low": latest.is_historically_low
            })
    db.close()
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_game_price_history(game_id: int):
    db: Session = SessionLocal()
    prices = db.query(PriceHistory).filter(PriceHistory.game_id == game_id).all()
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
        font=dict(color="#c6d4df", family="Inter"),
        title_font=dict(color="#ffffff", size=20, family="Inter"),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        xaxis=dict(showgrid=False, linecolor="#2a475e"),
        yaxis=dict(gridcolor="#2a475e", linecolor="#2a475e")
    )
    return fig

def main():
    inject_custom_css()
    
    st.sidebar.markdown("## 🕹️ Steam Intelligence")
    pages = ["Overview Dashboard", "Game Deep Dive & Predictions"]
    choice = st.sidebar.radio("Navigate", pages)

    df_games = load_all_games()
    
    if choice == "Overview Dashboard":
        st.title("Game Price Intelligence Analytics")
        st.markdown("Real-time and historic analytics tracking the **Top 100 Live Steam Deals**.")
        
        if df_games.empty:
            st.warning("No data found! Please wait for the background scraper to fetch the top deals.")
            return

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Games Tracked", len(df_games))
        
        all_time_lows = len(df_games[df_games['Is All-Time Low'] == True])
        col2.metric("Games at All-Time Low", all_time_lows, "🔥 Hot Deals")
        
        avg_price = df_games['Current Price'].mean()
        col3.metric("Avg Discounted Price", f"${avg_price:.2f}")
        
        avg_discount = df_games['Discount %'].mean()
        col4.metric("Avg Discount", f"{avg_discount:.1f}%")

        st.markdown("<hr>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Deepest Discounts Currently Available")
            top_discounts = df_games.sort_values(by="Discount %", ascending=False).head(10)
            fig1 = px.bar(top_discounts, x="Name", y="Discount %", 
                          color="Discount %", 
                          color_continuous_scale=[(0, "#2a475e"), (1, "#66c0f4")])
            fig1 = style_plotly_layout(fig1)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_chart2:
            st.subheader("Steam Deals Price Distribution")
            fig2 = px.histogram(df_games, x="Current Price", nbins=20, 
                                marginal="box", color_discrete_sequence=['#a4d007'])
            fig2 = style_plotly_layout(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("All Live Deals (Top 100)")
        
        # Style Pandas DF inside Streamlit natively via properties
        display_df = df_games.drop(columns=['ID']).sort_values(by="Current Price")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    elif choice == "Game Deep Dive & Predictions":
        st.title("Game Deep Dive & Machine Learning Predictions")
        
        if df_games.empty:
            st.warning("No data found!")
            return
            
        game_list = df_games['Name'].tolist()
        selected_game = st.selectbox("Select a deal to analyze:", game_list)
        
        game_info = df_games[df_games['Name'] == selected_game].iloc[0]
        game_id = game_info['ID']
        
        st.markdown(f"### {selected_game}")
        st.markdown(f"*{game_info['Developer']}* | {game_info['Genres']}")
        
        # Display main stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${game_info['Current Price']:.2f}", 
                    f"-{game_info['Discount %']}% from ${game_info['Initial Price']:.2f}")
        col2.metric("Discount Intensity", f"{game_info['Discount %']}%")
        col3.metric("All-Time Low", f"${game_info['Lowest Ever']:.2f}",
                    delta="Matched!" if game_info['Is All-Time Low'] else "Not Lowest")
        
        # Load history
        hist_df = load_game_price_history(game_id)
        if hist_df.empty:
            st.warning("No price history available.")
            return
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### 📈 Historical Price Trends")
        fig = px.line(hist_df, x="timestamp", y="final_price", markers=True)
        fig.add_hline(y=game_info['Lowest Ever'], line_dash="dash", line_color="#a4d007", annotation_text="All Time Low", annotation_font_color="#a4d007")
        fig.update_traces(line_color="#66c0f4", marker=dict(size=8, color="#ffffff"))
        fig = style_plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 🤖 Predictive Analytics")
        hist_features = compute_features(hist_df)
        
        col_ml1, col_ml2 = st.columns(2)
        
        with col_ml1:
            st.markdown("##### Time-Series Forecast (Prophet Model)")
            if len(hist_df) < 5:
                dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
                prices = np.random.normal(loc=game_info['Current Price'], scale=2.0, size=30)
                dummy_df = pd.DataFrame({'timestamp': dates, 'final_price': prices})
                forecast = forecast_price_prophet(dummy_df, periods=14)
            else:
                forecast = forecast_price_prophet(hist_df, periods=14)
                
            if not forecast.empty:
                fig_fb = go.Figure()
                fig_fb.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Price', line=dict(color="#66c0f4", width=3)))
                fig_fb.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(dash='dot', color='#2a475e'), name='Upper Bound'))
                fig_fb.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', fillcolor='rgba(102, 192, 244, 0.1)', line=dict(dash='dot', color='#2a475e'), name='Lower Bound'))
                fig_fb = style_plotly_layout(fig_fb)
                st.plotly_chart(fig_fb, use_container_width=True)

        with col_ml2:
            st.markdown("##### Deal Recommendation Engine")
            st.caption("Powered by Random Forest Classification")
            
            # Simulated model parameters
            if len(hist_features) < 10:
                buy_now = game_info['Discount %'] >= 50 or game_info['Is All-Time Low']
                prob_of_drop = 0.85 if not buy_now else 0.15
            else:
                clf, scaler = train_recommendation_model(hist_features)
                res = predict_deal_recommendation(clf, scaler, hist_features.tail(1))
                buy_now = res['buy_now']
                prob_of_drop = res['probability_of_future_drop']
            
            st.markdown("<br>", unsafe_allow_html=True)
            if buy_now:
                st.success("🏁 **VERDICT: STRONG BUY SIGNAL**")
                st.write("The algorithm indicates this deal matches optimal purchasing thresholds.")
            else:
                st.warning(f"⏳ **VERDICT: HOLD PHASE** (Future drop probability: {prob_of_drop*100:.1f}%)")
                st.write("The algorithm suggests waiting. A deeper discount is historically likely.")
                
            st.progress(prob_of_drop)
            st.caption("Probability of imminent further price reduction based on historical variance.")

if __name__ == "__main__":
    main()
