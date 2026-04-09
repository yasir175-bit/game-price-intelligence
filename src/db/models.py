from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
import datetime
from .database import Base

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    app_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    developer = Column(String(255), nullable=True)
    publisher = Column(String(255), nullable=True)
    genres = Column(String(500), nullable=True) # Comma separated
    release_date = Column(String(50), nullable=True)

    prices = relationship("PriceHistory", back_populates="game")

class PriceHistory(Base):
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    currency = Column(String(10), default="USD")
    initial_price = Column(Float, nullable=False) # Price before discount
    final_price = Column(Float, nullable=False) # Current price
    discount_percent = Column(Integer, default=0)
    
    # Feature Engineering fields (can be updated later or computed on fly)
    lowest_price_ever = Column(Float, nullable=True)
    is_historically_low = Column(Boolean, default=False)
    
    game = relationship("Game", back_populates="prices")
