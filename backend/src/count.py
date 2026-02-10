import chess.pgn

game_count = 0
with open("data/qwerty592_games.pgn") as pgn:
    while chess.pgn.read_headers(pgn):
        game_count += 1

print(f"Actual games found: {game_count}")