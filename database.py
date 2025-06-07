import sqlite3
import json
from datetime import datetime # Added for default timestamp if not provided

DB_NAME = "analysis_history.db"

def init_db():
    """Initializes the database and creates the analysis_history table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT,
                image_path TEXT,
                classification_type TEXT,
                confidence_score REAL,
                copy_move_score REAL,
                splicing_score REAL,
                output_dir TEXT,
                full_results_json TEXT
            )
        """)
        conn.commit()
        print("Database initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()

def save_analysis(timestamp, image_name, image_path,
                  classification_type, confidence_score, copy_move_score, splicing_score,
                  output_dir, full_results_dict):
    """Saves a new analysis record to the database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Ensure full_results_dict is converted to JSON string
        # Handle cases where it might not be serializable directly (e.g. complex objects)
        try:
            results_json = json.dumps(full_results_dict, default=lambda o: '<not serializable>')
        except TypeError:
            results_json = json.dumps({"error": "Failed to serialize full_results_dict due to complex objects."}, default=str)

        cursor.execute("""
            INSERT INTO analysis_history
            (timestamp, image_name, image_path, classification_type, confidence_score, copy_move_score, splicing_score, output_dir, full_results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, image_name, image_path, classification_type, confidence_score, copy_move_score, splicing_score, output_dir, results_json))
        conn.commit()
        print(f"Analysis for {image_name} saved to database.")
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"Error saving analysis to database: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_all_analyses():
    """Queries and returns all records from analysis_history, ordered by timestamp descending."""
    records = []
    try:
        conn = sqlite3.connect(DB_NAME)
        # To return dictionaries instead of tuples
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history ORDER BY timestamp DESC")
        records = cursor.fetchall()
        print(f"Retrieved {len(records)} records from database.")
    except sqlite3.Error as e:
        print(f"Error fetching analyses from database: {e}")
    finally:
        if conn:
            conn.close()
    # Convert sqlite3.Row objects to simple dicts if needed by QTableWidget,
    # though direct access like record['column_name'] is usually fine.
    return [dict(row) for row in records]


if __name__ == '__main__':
    # Example usage:
    print(f"Initializing database: {DB_NAME}")
    init_db()

    # Dummy data for testing
    dummy_results = {
        "classification": {
            "type": "Authentic",
            "confidence": 0.95,
            "copy_move_score": 10.0,
            "splicing_score": 5.0,
            "details": ["Low noise inconsistency", "Consistent JPEG quantization"]
        },
        "ela_mean": 10.5,
        # ... other analysis data ...
    }

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Test save
    # save_analysis(ts, "test_image.jpg", "/path/to/test_image.jpg",
    #               "Authentic", 0.95, 10.0, 5.0,
    #               "./gui_analysis_results/test_image", dummy_results)

    # Test get
    # all_records = get_all_analyses()
    # for record in all_records:
    #     print(f"ID: {record['id']}, Time: {record['timestamp']}, Image: {record['image_name']}, Class: {record['classification_type']}")
    #     # full_data = json.loads(record['full_results_json']) # To test deserialization
    #     # print(f"  Full data confidence: {full_data.get('classification',{}).get('confidence')}")

    print("Database script basic execution finished.")
