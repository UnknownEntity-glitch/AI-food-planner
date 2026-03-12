import sqlite3
from datetime import datetime
from typing import Optional, Dict, List, Any

class Database:
    def __init__(self, db_path: str = "data/bot.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER UNIQUE,
                goal TEXT,
                diet TEXT,
                allergies TEXT,
                age INTEGER,
                weight REAL,
                height REAL,
                gender TEXT,
                activity TEXT,
                calories_target REAL,
                protein_target REAL,
                fat_target REAL,
                carbs_target REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ingredient TEXT,
                quantity REAL,
                unit TEXT,
                raw_text TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meal_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                plan_date DATE,
                plan_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        self.conn.commit()

    def get_user(self, telegram_id: int) -> Optional[Dict]:
        cur = self.conn.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def create_user(self, telegram_id: int, goal: str, diet: str, allergies: str,
                    age: int, weight: float, height: float, gender: str, activity: str,
                    calories_target: float, protein_target: float, fat_target: float, carbs_target: float):
        self.conn.execute("""
            INSERT INTO users 
            (telegram_id, goal, diet, allergies, age, weight, height, gender, activity,
             calories_target, protein_target, fat_target, carbs_target)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (telegram_id, goal, diet, allergies, age, weight, height, gender, activity,
              calories_target, protein_target, fat_target, carbs_target))
        self.conn.commit()

    def update_user(self, telegram_id: int, **kwargs):
        fields = []
        values = []
        for key, value in kwargs.items():
            if value is not None:
                fields.append(f"{key} = ?")
                values.append(value)
        if fields:
            values.append(telegram_id)
            self.conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE telegram_id = ?", values)
            self.conn.commit()

    def add_inventory_item(self, user_id: int, ingredient: str, quantity: float, unit: str, raw_text: str):
        self.conn.execute("""
            INSERT INTO inventory (user_id, ingredient, quantity, unit, raw_text)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, ingredient, quantity, unit, raw_text))
        self.conn.commit()

    def get_inventory(self, user_id: int) -> List[Dict]:
        cur = self.conn.execute("""
            SELECT ingredient, quantity, unit, raw_text
            FROM inventory
            WHERE user_id = ?
            ORDER BY ingredient
        """, (user_id,))
        return [dict(row) for row in cur.fetchall()]

    def clear_inventory(self, user_id: int):
        self.conn.execute("DELETE FROM inventory WHERE user_id = ?", (user_id,))
        self.conn.commit()

    def remove_inventory_item(self, user_id: int, raw_text: str):
        self.conn.execute("DELETE FROM inventory WHERE user_id = ? AND raw_text = ?", (user_id, raw_text))
        self.conn.commit()

    def add_chat_message(self, user_id: int, role: str, content: str):
        self.conn.execute("""
            INSERT INTO chat_history (user_id, role, content)
            VALUES (?, ?, ?)
        """, (user_id, role, content))
        self.conn.commit()

    def get_chat_history(self, user_id: int, limit: int = 10) -> List[tuple]:
        cur = self.conn.execute("""
            SELECT role, content FROM chat_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cur.fetchall()
        return [(row[0], row[1]) for row in rows][::-1]

    def save_meal_plan(self, user_id: int, plan_date: str, plan_json: str):
        self.conn.execute("""
            INSERT INTO meal_plans (user_id, plan_date, plan_json)
            VALUES (?, ?, ?)
        """, (user_id, plan_date, plan_json))
        self.conn.commit()