import chess.pgn
import torch
import numpy as np
import os

MY_USERNAME = "qwerty592" 
INPUT_PGN = "data/qwerty592_games.pgn"
OUTPUT_FILE = "data/training_data.pt"

def board_to_tensor(board):
    """Converts a chess board object into a 13x8x8 tensor."""
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        plane_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[plane_idx, row, col] = 1.0
        
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    return tensor

def move_to_index(move):
    """Maps a move to an index in the range [0, 4095]."""
    return move.from_square * 64 + move.to_square

def augment_data(tensor, move_idx):
    """
    Returns a horizontally flipped version of the board tensor 
    and the corresponding flipped move index.
    """
    flipped_tensor = np.flip(tensor, axis=2).copy()

    from_sq = move_idx // 64
    to_sq = move_idx % 64
    
    f_row, f_col = divmod(from_sq, 8)
    t_row, t_col = divmod(to_sq, 8)
    
    new_f_col = 7 - f_col
    new_t_col = 7 - t_col
    
    new_from_sq = f_row * 8 + new_f_col
    new_to_sq = t_row * 8 + new_t_col
    new_move_idx = new_from_sq * 64 + new_to_sq
    
    return flipped_tensor, new_move_idx

def preprocess_games():
    X = []
    y = []
    
    if not os.path.exists(INPUT_PGN):
        print(f"Error: {INPUT_PGN} not found.")
        return

    pgn = open(INPUT_PGN)
    game_count = 0
    
    print("Starting preprocessing with data augmentation...")

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None: 
            break
        
        # Determine player color
        white_player = game.headers.get("White", "")
        black_player = game.headers.get("Black", "")
        
        if white_player == MY_USERNAME:
            my_color = chess.WHITE
        elif black_player == MY_USERNAME:
            my_color = chess.BLACK
        else:
            continue

        board = game.board()
        for move in game.mainline_moves():
            if board.turn == my_color:
                current_tensor = board_to_tensor(board)
                current_move_idx = move_to_index(move)
                X.append(current_tensor)
                y.append(current_move_idx)
                
                flipped_tensor, flipped_move_idx = augment_data(current_tensor, current_move_idx)
                X.append(flipped_tensor)
                y.append(flipped_move_idx)
            
            board.push(move)
            
        game_count += 1
        if game_count % 100 == 0:
            print(f"Processed {game_count} games... Current dataset size: {len(X)}")

    print("Converting to tensors (this may take a moment)...")
    X_tensor = torch.from_numpy(np.array(X))
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    torch.save({'positions': X_tensor, 'moves': y_tensor}, OUTPUT_FILE)
    print(f"Finished! Saved {len(X)} positions (Original + Augmented) to {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_games()