import torch
import chess
import numpy as np
from model import ChessStyleBot
from preprocess import board_to_tensor

def load_trained_model(model_path, device):
    model = ChessStyleBot()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def get_bot_move(board, model, device):
    """
    Takes a python-chess Board object and returns a chess.Move object
    representing the model's highest-scored legal move.
    """

    board_np = board_to_tensor(board)
    board_tensor = torch.from_numpy(board_np)
    board_tensor = board_tensor.unsqueeze(0)
    board_tensor = board_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(board_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    legal_moves = list(board.legal_moves)
    legal_indices = []
    
    for move in legal_moves:
        idx = move.from_square * 64 + move.to_square
        legal_indices.append(idx)
    
    mask = np.zeros(4096)
    mask[legal_indices] = 1
    
    masked_probs = probabilities * mask
    
    if np.sum(masked_probs) == 0:
        return np.random.choice(legal_moves)

    best_move_idx = np.argmax(masked_probs)
    
    from_square = best_move_idx // 64
    to_square = best_move_idx % 64
    
    move = chess.Move(from_square, to_square)
    if board.piece_at(from_square).piece_type == chess.PAWN:
        if (chess.square_rank(to_square) == 7 and board.turn == chess.WHITE) or \
           (chess.square_rank(to_square) == 0 and board.turn == chess.BLACK):
            move.promotion = chess.QUEEN
            
    return move

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model("models/my_style_bot.pth", device)
    board = chess.Board()

    print("Bot is ready. You play White (use UCI like 'e2e4').")
    
    while not board.is_game_over():
        print("\n", board)
        if board.turn == chess.WHITE:
            move_str = input("\nYour move: ")
            try:
                board.push_uci(move_str)
            except:
                print("Invalid UCI move.")
                continue
        else:
            print("Bot is thinking...")
            move = get_bot_move(board, model, device)
            print(f"Bot plays: {move}")
            board.push(move)
            
    print("Game Over. Result:", board.result())