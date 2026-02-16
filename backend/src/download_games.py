import os
from pathlib import Path
import requests
import time
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

USERNAME = os.environ.get("CHESS_USERNAME", "qwerty592")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
OUTPUT_FILE = f"{USERNAME}_games.pgn"

headers = {
    'User-Agent': f'ChessBotProject/1.0 (Contact: {CONTACT_EMAIL})'
}

def download_all_games():
    archive_url = f"https://api.chess.com/pub/player/{USERNAME}/games/archives"
    print(f"Fetching archives from: {archive_url}")
    
    response = requests.get(archive_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching archives: {response.status_code}")
        return

    archives = response.json().get('archives', [])
    print(f"Found {len(archives)} months of games. Starting download...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, month_url in enumerate(archives):
            pgn_url = f"{month_url}/pgn"
            print(f"[{i+1}/{len(archives)}] Downloading: {month_url.split('/')[-2]}-{month_url.split('/')[-1]}")
            
            pgn_response = requests.get(pgn_url, headers=headers)
            
            if pgn_response.status_code == 200:
                f.write(pgn_response.text)
                f.write("\n\n")
            else:
                print(f"Failed to download {pgn_url}: {pgn_response.status_code}")
            
            time.sleep(0.5)

    print(f"\nSuccess! All games saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    download_all_games()