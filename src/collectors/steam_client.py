import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteamAPIClient:
    def __init__(self):
        self.store_api_url = "https://store.steampowered.com/api/appdetails"
        self.steamspy_api_url = "https://steamspy.com/api.php"

    def get_top_100_games(self):
        """Fetches the top 100 games played in the last 2 weeks using SteamSpy."""
        logger.info("Fetching Top 100 games from SteamSpy...")
        try:
            response = requests.get(f"{self.steamspy_api_url}?request=top100in2weeks", timeout=10)
            response.raise_for_status()
            data = response.json()
            # Returns a dict where keys are app_ids and values are game details
            app_ids = list(data.keys())
            return app_ids
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching top games: {e}")
            return []

    def get_app_details(self, app_id: str):
        """Fetches current price and metadata from the official Steam Store API."""
        try:
            # Store API is rate limited to 200 requests per 5 minutes or similar. 
            # We add a small delay to avoid spamming.
            response = requests.get(f"{self.store_api_url}?appids={app_id}", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and str(app_id) in data and data[str(app_id)]['success']:
                return data[str(app_id)]['data']
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching app details for {app_id}: {e}")
            return None

    def get_historical_price_cheapshark(self, steam_app_id: str):
        """
        Uses CheapShark public API as a fallback to SteamDB to get 'cheapest price ever'.
        Returns a dict: {'price': float, 'date': int_timestamp}
        """
        try:
            url = f"https://www.cheapshark.com/api/1.0/games?steamAppID={steam_app_id}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    cheapest = data[0].get("cheapestPriceEver", {})
                    return {
                        "price": float(cheapest.get("price", 0)),
                        "date": cheapest.get("date", 0)
                    }
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch CheapShark data for {steam_app_id}: {e}")
            return None
