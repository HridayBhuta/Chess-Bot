# Chess Bot — Backend (Training Pipeline)

Python pipeline that downloads your Chess.com games, trains a neural network on your playing style, and exports it to ONNX for browser inference.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/download_games.py` | Download all games from Chess.com API |
| 2 | `src/preprocess.py` | Parse PGN → 13×8×8 tensors + move labels |
| 3 | `src/training.py` | Train residual CNN with cross-entropy loss |
| 4 | `src/inference.py` | Play against the bot in the terminal |
| — | `src/count.py` | Count games in PGN file |

## Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download your games (edit username in download_games.py)
python -m src.download_games

# 2. Preprocess into training tensors
python -m src.preprocess

# 3. Train the model
python -m src.training

# 4. Play against the bot in the terminal
python -m src.inference
```

## Model Architecture

- **Input**: 13 × 8 × 8 tensor (12 piece planes + 1 turn plane)
- **Backbone**: 10 residual blocks, 128 filters, batch normalization, ReLU
- **Policy head**: Conv → flatten → FC → 4096 logits (64 × 64 source-target squares)
- **Training**: Cross-entropy loss, Adam optimizer, StepLR scheduling, 90/10 train/val split

## Output

After training, model files are saved to `models/`:

| File | Description |
|------|-------------|
| `my_style_bot.pth` | PyTorch checkpoint (gitignored) |
| `my_style_bot.onnx` | ONNX model for browser inference |
| `my_style_bot.onnx.data` | ONNX external data |

Copy the ONNX model to `frontend/public/models/` for the web app:

```bash
cp models/my_style_bot.onnx ../frontend/public/models/
```

## Dependencies

- Python 3.10+
- PyTorch
- python-chess
- NumPy
- Requests
- ONNX
