'use client';

import { useState, useEffect, useCallback } from 'react';
import { Chessboard } from 'react-chessboard';
import { Chess, Square } from 'chess.js';
import { loadModel, getBotMove } from '@/lib/chessBot';

export default function ChessGame() {
  const [game, setGame] = useState(new Chess());
  const [status, setStatus] = useState('Loading model...');
  const [isThinking, setIsThinking] = useState(false);
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [modelLoaded, setModelLoaded] = useState(false);

  useEffect(() => {
    loadModel()
      .then(() => {
        setModelLoaded(true);
        setStatus('Your turn');
      })
      .catch((err) => {
        setStatus('Failed to load model');
        console.error(err);
      });
  }, []);

  const makeBotMove = useCallback(async (currentGame: Chess) => {
    if (currentGame.isGameOver()) return;

    setIsThinking(true);
    setStatus('Thinking...');

    await new Promise((r) => setTimeout(r, 200));

    const botMove = await getBotMove(currentGame);

    if (botMove) {
      try {
        currentGame.move(botMove);
        setGame(new Chess(currentGame.fen()));
        setMoveHistory((prev) => [...prev, botMove]);

        if (currentGame.isGameOver()) {
          setStatus(
            currentGame.isCheckmate()
              ? 'Checkmate! Bot wins!'
              : currentGame.isDraw()
              ? 'Draw!'
              : 'Game over'
          );
        } else {
          setStatus('Your turn');
        }
      } catch {
        setStatus('Your turn');
      }
    }

    setIsThinking(false);
  }, []);

  function onDrop(sourceSquare: Square, targetSquare: Square): boolean {
    if (!modelLoaded || isThinking || game.turn() !== 'w') return false;

    try {
      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });
      if (!move) return false;

      const newGame = new Chess(game.fen());
      setGame(newGame);
      setMoveHistory((prev) => [...prev, move.san]);

      if (newGame.isGameOver()) {
        setStatus(
          newGame.isCheckmate()
            ? 'Checkmate! You win!'
            : newGame.isDraw()
            ? 'Draw!'
            : 'Game over'
        );
        return true;
      }

      setTimeout(() => makeBotMove(newGame), 100);
      return true;
    } catch {
      return false;
    }
  }

  function resetGame() {
    setGame(new Chess());
    setMoveHistory([]);
    setStatus('Your turn');
  }

  // Build paired move rows for display
  const moveRows: { num: number; white: string; black?: string }[] = [];
  for (let i = 0; i < moveHistory.length; i += 2) {
    moveRows.push({
      num: Math.floor(i / 2) + 1,
      white: moveHistory[i],
      black: moveHistory[i + 1],
    });
  }

  return (
    <div className="game-wrapper">
      <div className="board-area">
        <Chessboard
          position={game.fen()}
          onPieceDrop={onDrop}
          boardWidth={480}
          customBoardStyle={{
            borderRadius: '4px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
          }}
          customDarkSquareStyle={{ backgroundColor: '#b58863' }}
          customLightSquareStyle={{ backgroundColor: '#f0d9b5' }}
        />
      </div>

      <div className="side-panel">
        <div className="status-bar">
          <span className={isThinking ? 'pulse' : ''}>{status}</span>
        </div>

        <div className="move-list">
          <table>
            <tbody>
              {moveRows.map((row) => (
                <tr key={row.num}>
                  <td className="move-num">{row.num}.</td>
                  <td className="move-white">{row.white}</td>
                  <td className="move-black">{row.black ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <button onClick={resetGame} className="new-game-btn">
          New Game
        </button>
      </div>
    </div>
  );
}
