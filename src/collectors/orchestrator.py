import time
import logging
from sqlalchemy.orm import Session
from src.db.database import SessionLocal, init_db
from src.db.models import Game, PriceHistory
from src.collectors.steam_client import SteamAPIClient
from src.processing.cleaner import clean_game_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_game_data():
    """
    Main pipeline: Fetch Top 100 Deals, enrich with Steam API, clean data, and save to DB.
    """
    client = SteamAPIClient()
    db: Session = SessionLocal()

    deals = client.get_top_100_deals()
    if not deals:
        logger.error("Failed to fetch Top 100 Deals. Aborting sync.")
        return

    logger.info(f"Processing {len(deals)} games from active deals...")

    for i, deal in enumerate(deals):
        app_id = deal.get("steamAppID")
        if not app_id:
            continue
            
        logger.info(f"Fetching metadata for app_id: {app_id} ({i+1}/{len(deals)})")
        
        # 1. Fetch current Store Details purely for genres and developers
        raw_steam_data = client.get_app_details(str(app_id))
        if not raw_steam_data:
            time.sleep(1) # Rate limit protection
            continue

        # 2. Clean Data (we override pricing with CheapShark's guaranteed data)
        # We still use cleaner for genres, publishers
        cleaned_data = clean_game_data(raw_steam_data)
        if not cleaned_data:
            time.sleep(1)
            continue
            
        # Override pricing with the active deal data
        cleaned_data["initial_price"] = float(deal.get("normalPrice", 0))
        cleaned_data["final_price"] = float(deal.get("salePrice", 0))
        # CheapShark provides percentage as a string float e.g. "90.003"
        savings_str = deal.get("savings", "0")
        try:
            cleaned_data["discount_percent"] = int(float(savings_str))
        except:
            cleaned_data["discount_percent"] = 0
            
        # We don't want free games in our premium analytics
        if cleaned_data["initial_price"] == 0 and cleaned_data["final_price"] == 0:
            continue
            
        # 3. Check if we need to fetch all-time low (CheapShark fallback)
        game = db.query(Game).filter(Game.app_id == cleaned_data['app_id']).first()
        
        lowest_price = None
        if not game:
            logger.info(f"New game found {cleaned_data['name']}. Fetching historical low...")
            cs_data = client.get_historical_price_cheapshark(cleaned_data['app_id'])
            if cs_data:
                lowest_price = cs_data['price']
            
            game = Game(
                app_id=cleaned_data['app_id'],
                name=deal.get("title", cleaned_data['name']), # CheapShark title is sometimes cleaner
                developer=cleaned_data['developer'],
                publisher=cleaned_data['publisher'],
                genres=cleaned_data['genres'],
                release_date=cleaned_data['release_date']
            )
            db.add(game)
            db.commit()
            db.refresh(game)
        else:
            pass

        # 4. Insert Price History
        previous_prices = db.query(PriceHistory).filter(PriceHistory.game_id == game.id).all()
        cached_low = min([p.final_price for p in previous_prices]) if previous_prices else float('inf')
        
        if lowest_price is None:
            lowest_price = cached_low if cached_low != float('inf') else cleaned_data['final_price']
            
        is_historically_low = bool(cleaned_data['final_price'] <= lowest_price and cleaned_data['final_price'] > 0)

        price_entry = PriceHistory(
            game_id=game.id,
            currency=cleaned_data['currency'],
            initial_price=cleaned_data['initial_price'],
            final_price=cleaned_data['final_price'],
            discount_percent=cleaned_data['discount_percent'],
            lowest_price_ever=lowest_price,
            is_historically_low=is_historically_low
        )
        db.add(price_entry)
        db.commit()
        
        time.sleep(1.5)

    db.close()
    logger.info("Sync complete.")

if __name__ == "__main__":
    init_db()
    update_game_data()
