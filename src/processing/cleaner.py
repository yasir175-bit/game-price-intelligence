import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_game_data(raw_data: dict) -> dict:
    """
    Cleans raw Steam Store API data into a standardized dictionary for our database.
    Handles null prices, currencies, and extracts genres.
    """
    try:
        # Extract basic info
        name = raw_data.get('name', 'Unknown')
        app_id_str = str(raw_data.get('steam_appid'))
        
        developers = ", ".join(raw_data.get('developers', []))
        publishers = ", ".join(raw_data.get('publishers', []))
        
        genres_list = raw_data.get('genres', [])
        genres = ", ".join([g.get('description', '') for g in genres_list])
        
        release_date = raw_data.get('release_date', {}).get('date', '')

        # Extract pricing
        is_free = raw_data.get('is_free', False)
        price_overview = raw_data.get('price_overview', {})
        
        currency = price_overview.get('currency', 'INR')
        
        if is_free:
            initial_price = 0.0
            final_price = 0.0
            discount_percent = 0
        elif not price_overview:
            # Maybe upcoming or not for sale directly right now
            initial_price = -1.0
            final_price = -1.0
            discount_percent = 0
        else:
            # Steam returns price in cents (e.g., 999 for $9.99)
            initial_price = price_overview.get('initial', 0) / 100.0
            final_price = price_overview.get('final', 0) / 100.0
            discount_percent = price_overview.get('discount_percent', 0)

        cleaned = {
            "app_id": app_id_str,
            "name": name,
            "developer": developers,
            "publisher": publishers,
            "genres": genres,
            "release_date": release_date,
            "is_free": is_free,
            "currency": currency,
            "initial_price": initial_price,
            "final_price": final_price,
            "discount_percent": discount_percent
        }
        return cleaned

    except Exception as e:
        logger.error(f"Error cleaning game data: {e}")
        return None
