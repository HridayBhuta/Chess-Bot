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

def get_result_value(result_str, my_color):
    """Converts PGN result string to a value relative to the bot."""
    if result_str == "1/2-1/2":
        return 0.0
    if result_str == "1-0":
        return 3.0 if my_color == chess.WHITE else -2.0
    if result_str == "0-1":
        return 3.0 if my_color == chess.BLACK else -2.0
    return 0.0

def preprocess_games(input_pgn=INPUT_PGN, output_file=OUTPUT_FILE):
    X, y_policy, y_value = [], [], []
    
    if not os.path.exists(input_pgn):
        return

    pgn = open(input_pgn)
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None: break
        
        # Identify bot's color and game result
        white_player = game.headers.get("White", "")
        my_color = chess.WHITE if white_player == MY_USERNAME else chess.BLACK
        result_val = get_result_value(game.headers.get("Result", "*"), my_color)

        board = game.board()
        for move in game.mainline_moves():
            if board.turn == my_color:
                current_tensor = board_to_tensor(board)
                X.append(current_tensor)
                y_policy.append(move_to_index(move))
                y_value.append(result_val)
                
                # Flip for augmentation
                flipped_t, flipped_m = augment_data(current_tensor, move_to_index(move))
                X.append(flipped_t)
                y_policy.append(flipped_m)
                y_value.append(result_val)
            
            board.push(move)

    torch.save({
        'positions': torch.from_numpy(np.array(X)), 
        'moves': torch.tensor(y_policy, dtype=torch.long),
        'values': torch.tensor(y_value, dtype=torch.float32)
    }, output_file)

if __name__ == "__main__":
    preprocess_games()