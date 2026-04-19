"""
Database Manager for Disaster Prediction App
Uses SQLite via SQLAlchemy for persistent storage
"""

import sqlite3
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import inspect

Base = declarative_base()
DB_PATH = os.path.join(os.path.dirname(__file__), "disaster_predictions.db")


class EarthquakePrediction(Base):
    __tablename__ = "earthquake_predictions"
    id = Column(Integer, primary_key=True)
    latitude = Column(Float)
    longitude = Column(Float)
    depth = Column(Float)
    magnitude_pred = Column(Float)
    risk_level = Column(String(20))
    confidence = Column(Float)
    location_name = Column(String(200))
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_sent = Column(Boolean, default=False)


class FloodPrediction(Base):
    __tablename__ = "flood_predictions"
    id = Column(Integer, primary_key=True)
    region = Column(String(200))
    rainfall_mm = Column(Float)
    river_level = Column(Float)
    soil_moisture = Column(Float)
    flood_probability = Column(Float)
    risk_level = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)


class CyclonePrediction(Base):
    __tablename__ = "cyclone_predictions"
    id = Column(Integer, primary_key=True)
    lat = Column(Float)
    lon = Column(Float)
    wind_speed = Column(Float)
    pressure = Column(Float)
    category = Column(Integer)
    intensity_pred = Column(String(50))
    landfall_risk = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class WildfirePrediction(Base):
    __tablename__ = "wildfire_predictions"
    id = Column(Integer, primary_key=True)
    region = Column(String(200))
    temperature = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    drought_index = Column(Float)
    fire_probability = Column(Float)
    risk_level = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)


class TsunamiPrediction(Base):
    __tablename__ = "tsunami_predictions"
    id = Column(Integer, primary_key=True)
    origin_lat = Column(Float)
    origin_lon = Column(Float)
    earthquake_mag = Column(Float)
    depth = Column(Float)
    wave_height_pred = Column(Float)
    eta_minutes = Column(Float)
    affected_coasts = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DroughtPrediction(Base):
    __tablename__ = "drought_predictions"
    id = Column(Integer, primary_key=True)
    region = Column(String(200))
    spi_index = Column(Float)
    temperature_anomaly = Column(Float)
    precipitation_deficit = Column(Float)
    severity = Column(String(30))
    duration_weeks = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AlertLog(Base):
    __tablename__ = "alert_logs"
    id = Column(Integer, primary_key=True)
    disaster_type = Column(String(50))
    severity = Column(String(20))
    location = Column(String(200))
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)


class DBManager:
    def __init__(self):
        self.engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def save_earthquake(self, data: dict):
        record = EarthquakePrediction(**data)
        self.session.add(record)
        self.session.commit()
        return record.id

    def save_flood(self, data: dict):
        record = FloodPrediction(**data)
        self.session.add(record)
        self.session.commit()
        return record.id

    def save_cyclone(self, data: dict):
        record = CyclonePrediction(**data)
        self.session.add(record)
        self.session.commit()
        return record.id

    def save_wildfire(self, data: dict):
        record = WildfirePrediction(**data)
        self.session.add(record)
        self.session.commit()
        return record.id

    def save_tsunami(self, data: dict):
        record = TsunamiPrediction(**data)
        self.session.add(record)
        self.session.commit()
        return record.id

    def save_drought(self, data: dict):
        record = DroughtPrediction(**data)
        self.session.add(record)
        self.session.commit()
        return record.id

    def log_alert(self, data: dict):
        record = AlertLog(**data)
        self.session.add(record)
        self.session.commit()
        return record.id

    def get_recent_predictions(self, table_name, limit=20):
        mapping = {
            "earthquake": EarthquakePrediction,
            "flood": FloodPrediction,
            "cyclone": CyclonePrediction,
            "wildfire": WildfirePrediction,
            "tsunami": TsunamiPrediction,
            "drought": DroughtPrediction,
        }
        model = mapping.get(table_name)
        if model:
            rows = self.session.query(model).order_by(model.timestamp.desc()).limit(limit).all()
            return [r.__dict__ for r in rows]
        return []

    def get_alert_logs(self, limit=50):
        rows = self.session.query(AlertLog).order_by(AlertLog.timestamp.desc()).limit(limit).all()
        return [r.__dict__ for r in rows]

    def get_stats(self):
        stats = {}
        for name, model in [
            ("earthquake", EarthquakePrediction),
            ("flood", FloodPrediction),
            ("cyclone", CyclonePrediction),
            ("wildfire", WildfirePrediction),
            ("tsunami", TsunamiPrediction),
            ("drought", DroughtPrediction),
        ]:
            stats[name] = self.session.query(model).count()
        stats["alerts"] = self.session.query(AlertLog).count()
        return stats

    def close(self):
        self.session.close()
