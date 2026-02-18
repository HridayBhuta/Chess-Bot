import os
import threading
import chess
import numpy as np
import torch
import torch.nn.functional as F

from model import ChessStyleBot
from preprocess import board_to_tensor, move_to_index

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PTH_PATH = os.path.join(MODEL_DIR, "my_style_bot.pth")
ONNX_PATH = os.path.join(MODEL_DIR, "my_style_bot.onnx")
FRONTEND_ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "frontend", "public", "models", "my_style_bot.onnx"
)

_train_lock = threading.Lock()

RL_LEARNING_RATE = 1e-5
DISCOUNT_FACTOR = 0.99


def _compute_reward(result: str, bot_color: str) -> float:
    """
    Map game result to a scalar reward for the bot.
    result: '1-0', '0-1', '1/2-1/2'
    bot_color: 'black' (the bot always plays black in the current setup)
    """
    if result == "1/2-1/2":
        return 0.0
    bot_won = (result == "1-0" and bot_color == "white") or (
        result == "0-1" and bot_color == "black"
    )
    return 2.0 if bot_won else -1.0


def _export_onnx(model: ChessStyleBot, device: torch.device) -> None:
    """Re-export the updated model to ONNX for browser inference."""
    import shutil

    model.eval()
    dummy = torch.randn(1, 13, 8, 8, device=device)
    # Use legacy exporter (dynamo=False) to produce a single self-contained .onnx file
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )

    os.makedirs(os.path.dirname(FRONTEND_ONNX_PATH), exist_ok=True)
    shutil.copy2(ONNX_PATH, FRONTEND_ONNX_PATH)


def train_on_game(moves_uci: list[str], result: str, bot_color: str = "black") -> dict:
    """
    Run one REINFORCE update using the positions where the bot moved.

    Parameters
    ----------
    moves_uci : list of UCI strings for the full game, e.g. ["e2e4", "e7e5", ...]
    result    : game outcome — "1-0", "0-1", or "1/2-1/2"
    bot_color : which side the bot played ("white" or "black")

    Returns
    -------
    dict with training stats
    """
    reward = _compute_reward(result, bot_color)
    if reward == 0.0:
        return {"status": "skipped", "reason": "draw — no gradient update"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with _train_lock:
        model = ChessStyleBot()
        model.load_state_dict(torch.load(PTH_PATH, map_location=device))
        model.to(device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=RL_LEARNING_RATE)

        bot_is_white = bot_color == "white"
        board = chess.Board()

        states: list[np.ndarray] = []
        actions: list[int] = []

        for uci in moves_uci:
            move = chess.Move.from_uci(uci)
            is_bot_turn = (board.turn == chess.WHITE) == bot_is_white
            if is_bot_turn:
                states.append(board_to_tensor(board))
                actions.append(move_to_index(move))
            board.push(move)

        if not states:
            return {"status": "skipped", "reason": "no bot moves found"}

        state_t = torch.from_numpy(np.array(states)).to(device)
        action_t = torch.tensor(actions, dtype=torch.long, device=device)

        n = len(actions)
        discounts = np.array([DISCOUNT_FACTOR ** (n - 1 - i) for i in range(n)], dtype=np.float32)
        reward_t = torch.from_numpy(discounts * reward).to(device)

        logits = model(state_t)
        log_probs = F.log_softmax(logits, dim=1)
        chosen_log_probs = log_probs.gather(1, action_t.unsqueeze(1)).squeeze(1)

        loss = -(reward_t * chosen_log_probs).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        torch.save(model.state_dict(), PTH_PATH)

        _export_onnx(model, device)

    return {
        "status": "updated",
        "reward": reward,
        "positions_trained": n,
        "loss": round(loss.item(), 6),
    }
