# app/db_handler.py
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

DB_CONFIG = {
    "dbname": "finance_db",
    "user": "postgres",
    "password": "",  # the one you used with psql
    "host": "localhost",
    "port": 5432
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def initialize_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            date TIMESTAMP,
            merchant TEXT,
            amount NUMERIC,
            mode TEXT,
            name TEXT,
            category TEXT,
            source TEXT,
            raw_text TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_transaction(date, merchant, amount, mode, name, category, source="receipt", raw_text=""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO transactions (date, merchant, amount, mode, name, category, source, raw_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        date or datetime.now(),
        merchant,
        amount,
        mode,
        name,
        category,
        source,
        raw_text
    ))
    conn.commit()
    cur.close()
    conn.close()
