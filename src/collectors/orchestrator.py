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
    Main pipeline: Fetch Top 100 games, clean data, and save to DB.
    """
    client = SteamAPIClient()
    db: Session = SessionLocal()

    app_ids = client.get_top_100_games()
    if not app_ids:
        logger.error("Failed to fetch Top 100 app IDs. Aborting sync.")
        return

    logger.info(f"Processing {len(app_ids)} games...")

    # For testing or performance, we might just process a subset.
    # We will process all 100 since the user wants 100.
    for i, app_id in enumerate(app_ids):
        logger.info(f"Fetching details for app_id: {app_id} ({i+1}/{len(app_ids)})")
        
        # 1. Fetch current Store Details
        raw_data = client.get_app_details(str(app_id))
        if not raw_data:
            time.sleep(1) # Rate limit protection
            continue

        # 2. Clean Data
        cleaned_data = clean_game_data(raw_data)
        if not cleaned_data or cleaned_data['final_price'] == -1:
            time.sleep(1)
            continue
            
        # 3. Check if we need to fetch all-time low (CheapShark fallback)
        # We only do this if we haven't seen the game or on first run to save rate limits
        game = db.query(Game).filter(Game.app_id == cleaned_data['app_id']).first()
        
        lowest_price = None
        if not game:
            logger.info(f"New game found {cleaned_data['name']}. Fetching historical low...")
            cs_data = client.get_historical_price_cheapshark(cleaned_data['app_id'])
            if cs_data:
                lowest_price = cs_data['price']
            
            game = Game(
                app_id=cleaned_data['app_id'],
                name=cleaned_data['name'],
                developer=cleaned_data['developer'],
                publisher=cleaned_data['publisher'],
                genres=cleaned_data['genres'],
                release_date=cleaned_data['release_date']
            )
            db.add(game)
            db.commit()
            db.refresh(game)
        else:
            # We already have history, we can figure out all time low from our own DB
            # but for robustness let's just query CheapShark quickly if needed, or rely on our DB.
            pass

        # 4. Insert Price History
        # We calculate lowest_price_ever
        previous_prices = db.query(PriceHistory).filter(PriceHistory.game_id == game.id).all()
        cached_low = min([p.final_price for p in previous_prices]) if previous_prices else float('inf')
        
        if lowest_price is None:
            # If we didn't fetch from CheapShark, use our cached low or current price
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
        
        # Steam API is rate limited to ~200 requests / 5 mins.
        # We add 1.5 second delay -> 40 requests/min -> well within limits.
        time.sleep(1.5)

    db.close()
    logger.info("Sync complete.")

if __name__ == "__main__":
    init_db()
    update_game_data()
