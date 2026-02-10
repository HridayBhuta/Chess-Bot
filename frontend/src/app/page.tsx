'use client';

import dynamic from 'next/dynamic';

const ChessGame = dynamic(() => import('@/components/ChessGame'), {
  ssr: false,
  loading: () => (
    <div className="loading">
      <div className="spinner"></div>
      <p>Loading...</p>
    </div>
  ),
});

export default function Home() {
  return (
    <main className="main">
      <h1>Chess Game</h1>
      <ChessGame />
    </main>
  );
}
