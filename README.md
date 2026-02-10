# Chess Bot

A chess bot trained on your personal [Chess.com](https://www.chess.com) games. It learns your playing style using a residual CNN and runs entirely in the browser via ONNX Runtime WebAssembly.

## Project Structure

```
Chess-Bot/
├── backend/                          # Python — training pipeline
│   ├── src/
│   │   ├── download_games.py         # Download games from Chess.com API
│   │   ├── preprocess.py             # Parse PGN → 13×8×8 tensors + move labels
│   │   ├── model.py                  # ChessStyleBot — 10-block residual CNN
│   │   ├── training.py               # Train with cross-entropy loss + LR scheduling
│   │   ├── inference.py              # CLI inference — play against the bot locally
│   │   └── count.py                  # Count games in PGN file
│   ├── models/                       # Trained model files
│   │   └── my_style_bot.onnx        # ONNX model for browser inference
│   ├── data/                         # Downloaded PGN + processed data (gitignored)
│   ├── requirements.txt
│   └── README.md
├── frontend/                         # Next.js web app
│   ├── public/
│   │   └── models/
│   │       └── my_style_bot.onnx     # Model served to the browser
│   ├── src/
│   │   ├── app/                      # Next.js App Router pages
│   │   ├── components/
│   │   │   └── ChessGame.tsx         # Chessboard + game logic
│   │   └── lib/
│   │       └── chessBot.ts           # ONNX inference in the browser
│   ├── package.json
│   └── README.md
└── README.md
```

## How It Works

1. **Download** — Fetches all your games from the Chess.com public API
2. **Preprocess** — Encodes each board position as a 13-channel 8×8 tensor (6 piece types × 2 colors + turn indicator) with move labels as indices into a 4096-dim policy vector
3. **Train** — A residual CNN (10 blocks, 128 channels) learns to predict your moves via cross-entropy loss
4. **Export** — The trained PyTorch model is converted to ONNX format
5. **Play** — The Next.js frontend loads the ONNX model in the browser using WebAssembly — no server needed

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### Train the Model

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_games
python -m src.preprocess
python -m src.training
```

### Run the Web App

```bash
cd frontend
npm install

# Copy WASM runtime files to public/
cp node_modules/onnxruntime-web/dist/ort-wasm*.wasm public/
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs public/
cp node_modules/onnxruntime-web/dist/ort.wasm.min.js public/

npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to play.

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for detailed instructions.

## Model Architecture

- **Input**: 13 × 8 × 8 tensor (12 piece planes + 1 turn plane)
- **Backbone**: 10 residual blocks with 128 filters, batch normalization, and ReLU
- **Policy head**: Conv → flatten → FC → 4096 logits (64 source squares × 64 target squares)
- **Move selection**: Masked softmax over legal moves, pick highest probability

## Tech Stack

| Component | Technology |
|-----------|------------|
| Training | Python, PyTorch |
| Model format | ONNX |
| Frontend | Next.js, React, TypeScript |
| Chess logic | chess.js |
| Board UI | react-chessboard |
| Browser inference | ONNX Runtime Web (WASM) |

## License

MIT
