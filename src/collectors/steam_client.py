import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteamAPIClient:
    def __init__(self):
        self.store_api_url = "https://store.steampowered.com/api/appdetails"
        self.steamspy_api_url = "https://steamspy.com/api.php"

    def get_top_250_deals(self):
        """Fetches the top 250 current deals for Steam via CheapShark to guarantee non-zero prices."""
        logger.info("Fetching Top 250 Deals from CheapShark...")
        try:
            # storeID=1 is Steam. sortBy=Deal Rating ensures high quality deals.
            url = "https://www.cheapshark.com/api/1.0/deals?storeID=1&pageSize=250&sortBy=Deal%20Rating"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Returns a list of deal objects
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching top deals: {e}")
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
