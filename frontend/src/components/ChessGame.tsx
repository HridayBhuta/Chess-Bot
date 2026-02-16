'use client';

import { useState, useEffect, useCallback } from 'react';
import { Chessboard } from 'react-chessboard';
import { Chess, Square } from 'chess.js';
import { loadModel, getBotMove } from '@/lib/chessBot';

const PIECE_SYMBOLS: Record<string, string> = {
  p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚',
};

export default function ChessGame() {
  const [game, setGame] = useState(new Chess());
  const [status, setStatus] = useState('Loading model...');
  const [isThinking, setIsThinking] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [capturedWhite, setCapturedWhite] = useState<string[]>([]);
  const [capturedBlack, setCapturedBlack] = useState<string[]>([]);

  useEffect(() => {
    loadModel()
      .then(() => {
        setModelLoaded(true);
        setStatus('Your Turn (White)');
      })
      .catch((err) => {
        setStatus('Failed to load model');
        console.error(err);
      });
  }, []);

  function updateCaptured(g: Chess) {
    const white: string[] = [];
    const black: string[] = [];
    for (const move of g.history({ verbose: true })) {
      if (move.captured) {
        const sym = PIECE_SYMBOLS[move.captured] || move.captured;
        if (move.color === 'w') black.push(sym);
        else white.push(sym);
      }
    }
    setCapturedWhite(white);
    setCapturedBlack(black);
  }

  const makeBotMove = useCallback(async (currentGame: Chess) => {
    if (currentGame.isGameOver()) return;

    setIsThinking(true);
    setStatus('AI is thinking...');

    await new Promise((r) => setTimeout(r, 200));

    const botMove = await getBotMove(currentGame);

    if (botMove) {
      try {
        currentGame.move(botMove);
        const newGame = new Chess(currentGame.fen());

        // Rebuild history for captured tracking
        setGame(newGame);
        updateCaptured(currentGame);

        if (currentGame.isGameOver()) {
          setStatus(
            currentGame.isCheckmate()
              ? 'Checkmate! Bot wins!'
              : currentGame.isDraw()
              ? 'Draw!'
              : 'Game over'
          );
        } else {
          setStatus('Your Turn (White)');
        }
      } catch {
        setStatus('Your Turn (White)');
      }
    }

    setIsThinking(false);
  }, []);

  function onDrop(sourceSquare: Square, targetSquare: Square): boolean {
    if (!modelLoaded || isThinking || game.turn() !== 'w') return false;

    try {
      const gameCopy = new Chess(game.fen());
      // Copy move history for captured tracking
      const prevHistory = game.history({ verbose: true });

      const move = gameCopy.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });
      if (!move) return false;

      setGame(new Chess(gameCopy.fen()));

      // Update captured pieces
      const white: string[] = [...capturedWhite];
      const black: string[] = [...capturedBlack];
      if (move.captured) {
        const sym = PIECE_SYMBOLS[move.captured] || move.captured;
        if (move.color === 'w') black.push(sym);
        else white.push(sym);
      }
      setCapturedWhite(white);
      setCapturedBlack(black);

      if (gameCopy.isGameOver()) {
        setStatus(
          gameCopy.isCheckmate()
            ? 'Checkmate! You win!'
            : gameCopy.isDraw()
            ? 'Draw!'
            : 'Game over'
        );
        return true;
      }

      setTimeout(() => makeBotMove(gameCopy), 100);
      return true;
    } catch {
      return false;
    }
  }

  function resetGame() {
    setGame(new Chess());
    setCapturedWhite([]);
    setCapturedBlack([]);
    setStatus('Your Turn (White)');
  }

  const turnIcon = game.turn() === 'w' ? '♟' : '♛';

  return (
    <div className="card">
      <div className="status-badge">
        <span className={isThinking ? 'pulse' : ''}>
          {turnIcon} {status}
        </span>
      </div>

      <div className="board-area">
        <Chessboard
          position={game.fen()}
          onPieceDrop={onDrop}
          boardWidth={560}
          customBoardStyle={{
            borderRadius: '4px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
          }}
          customDarkSquareStyle={{ backgroundColor: '#b58863' }}
          customLightSquareStyle={{ backgroundColor: '#f0d9b5' }}
        />
      </div>

      <div className="captured-row">
        <div className="captured-box">
          <span className="captured-label">♟ White Captured</span>
          <span className="captured-pieces">
            {capturedWhite.length > 0 ? capturedWhite.join(' ') : 'None'}
          </span>
        </div>
        <div className="captured-box">
          <span className="captured-label">♛ Black Captured</span>
          <span className="captured-pieces">
            {capturedBlack.length > 0 ? capturedBlack.join(' ') : 'None'}
          </span>
        </div>
      </div>

      <button onClick={resetGame} className="reset-btn">
        ↻ RESET BOARD
      </button>
    </div>
  );
}
