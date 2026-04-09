import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from apscheduler.schedulers.blocking import BlockingScheduler
import logging
from src.collectors.orchestrator import update_game_data
from src.db.database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Auto-Refresh Scheduler...")
    
    # Initialize the DB schema in case tables don't exist yet
    init_db()
    
    # Run once immediately on startup
    update_game_data()

    scheduler = BlockingScheduler()
    # Schedule to run every 6 hours
    scheduler.add_job(update_game_data, 'interval', hours=6)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
