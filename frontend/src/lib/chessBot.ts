import { Chess } from 'chess.js';

/* eslint-disable @typescript-eslint/no-explicit-any */

let session: any = null;
let loadingPromise: Promise<void> | null = null;
let ortModule: any = null;

function getOrt(): any {
  if (ortModule) return ortModule;
  ortModule = (window as any).ort;
  if (!ortModule) throw new Error('ONNX Runtime not loaded');
  return ortModule;
}

export async function loadModel(): Promise<void> {
  if (session) return;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    // Load the ONNX Runtime script dynamically
    if (!(window as any).ort) {
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement('script');
        script.src = '/ort.wasm.min.js';
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('Failed to load ONNX Runtime script'));
        document.head.appendChild(script);
      });
    }

    const ort = getOrt();
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = '/';

    const response = await fetch('/models/my_style_bot.onnx?t=' + Date.now());
    if (!response.ok) throw new Error(`Model fetch failed: ${response.status}`);
    const buf = await response.arrayBuffer();

    session = await ort.InferenceSession.create(buf, {
      executionProviders: ['wasm'],
    });
    console.log('Model loaded. Inputs:', session.inputNames, 'Outputs:', session.outputNames);
  })();

  try {
    await loadingPromise;
  } catch (e) {
    loadingPromise = null; // allow retry on failure
    throw e;
  }
}

// Convert chess.js board to 13x8x8 tensor matching Python preprocessing
function boardToTensor(chess: Chess): Float32Array {
  const tensor = new Float32Array(13 * 8 * 8).fill(0);

  const pieceTypeMap: Record<string, number> = {
    p: 0, n: 1, b: 2, r: 3, q: 4, k: 5,
  };

  const board = chess.board();

  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const piece = board[rank][file];
      if (piece) {
        const pieceType = pieceTypeMap[piece.type];
        const colorOffset = piece.color === 'w' ? 0 : 6;
        const planeIdx = pieceType + colorOffset;
        // chess.js board[0] = rank 8 (top). Convert to row where a1=square 0.
        const row = 7 - rank;
        const col = file;
        tensor[planeIdx * 64 + row * 8 + col] = 1.0;
      }
    }
  }

  // Turn plane (plane 12): all 1s if white to move
  if (chess.turn() === 'w') {
    for (let i = 0; i < 64; i++) {
      tensor[12 * 64 + i] = 1.0;
    }
  }

  return tensor;
}

export async function getBotMove(chess: Chess): Promise<string | null> {
  if (!session) await loadModel();
  if (!session) return null;

  const ort = getOrt();
  const inputData = boardToTensor(chess);
  const inputTensor = new ort.Tensor('float32', inputData, [1, 13, 8, 8]);

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const feeds: Record<string, any> = { [inputName]: inputTensor };
  const results = await session.run(feeds);
  const logits = results[outputName].data as Float32Array;

  // Softmax
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  const expArr = new Float64Array(logits.length);
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    expArr[i] = Math.exp(logits[i] - maxVal);
    sumExp += expArr[i];
  }

  // Score each legal move
  const legalMoves = chess.moves({ verbose: true });
  let bestMove: string | null = null;
  let bestScore = -1;

  for (const move of legalMoves) {
    const fromFile = move.from.charCodeAt(0) - 97;
    const fromRank = parseInt(move.from[1]) - 1;
    const fromIdx = fromRank * 8 + fromFile;

    const toFile = move.to.charCodeAt(0) - 97;
    const toRank = parseInt(move.to[1]) - 1;
    const toIdx = toRank * 8 + toFile;

    const moveIdx = fromIdx * 64 + toIdx;
    const score = expArr[moveIdx] / sumExp;

    if (score > bestScore) {
      bestScore = score;
      bestMove = move.san;
    }
  }

  return bestMove;
}

/**
 * Send completed game to the RL backend for a policy-gradient update.
 * Returns the server response (status, reward, loss, etc.).
 */
export async function submitGameForLearning(
  movesUci: string[],
  result: string,
  botColor: string = 'black'
): Promise<{ status: string; reward?: number; loss?: number; positions_trained?: number; reason?: string }> {
  const res = await fetch('/api/learn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves: movesUci, result, bot_color: botColor }),
  });
  if (!res.ok) throw new Error(`RL server error: ${res.status}`);
  return res.json();
}

/**
 * Force-reload the ONNX model so the browser picks up updated weights.
 */
export async function reloadModel(): Promise<void> {
  session = null;
  loadingPromise = null;
  await loadModel();
}
