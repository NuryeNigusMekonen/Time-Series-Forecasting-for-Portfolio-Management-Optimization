# main.py

from src.data_loader import fetch_and_save_all
from src.preprocessing import process_all_assets

def main():
    print("[INFO] Starting Time Series Portfolio Project")
    print("[STEP] Preprocessing and feature engineering...")
    process_all_assets()

    print("[INFO] Setup complete.")

if __name__ == "__main__":
    main()
