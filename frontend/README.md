# Chess Bot — Frontend

A web-based chess game where you play against an AI bot trained on your playing style. Runs entirely in the browser using ONNX Runtime WebAssembly.

## Features

- Visual chessboard with drag-and-drop piece movement
- AI opponent running entirely in the browser using ONNX Runtime
- Move history tracking
- Responsive design

## Getting Started

### Install dependencies

```bash
cd frontend
npm install
```

### Run development server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to play.

### Build for production

```bash
npm run build
```

## Deploy to Vercel

### Option 1: Vercel CLI

```bash
npm i -g vercel
cd frontend
vercel
```

### Option 2: GitHub Integration

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Import your repository
4. Set the root directory to `frontend`
5. Deploy!

## Project Structure

```
frontend/
├── public/
│   └── models/
│       └── my_style_bot.onnx    # Your trained model
├── src/
│   ├── app/
│   │   ├── layout.tsx           # Root layout
│   │   ├── page.tsx             # Main page
│   │   └── globals.css          # Styles
│   ├── components/
│   │   └── ChessGame.tsx        # Chess board component
│   └── lib/
│       └── chessBot.ts          # ONNX inference logic
└── package.json
```

## How It Works

The chess bot runs entirely in your browser using WebAssembly (WASM). When you make a move:

1. The board state is converted to a tensor format
2. The ONNX model processes the position
3. Legal moves are filtered and scored
4. The best move is selected and played

No server-side computation needed - everything runs client-side!
