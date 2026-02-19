import { NextRequest, NextResponse } from 'next/server';

const RL_SERVER = process.env.RL_SERVER || 'http://localhost:8000';

export async function POST(req: NextRequest) {
  const body = await req.json();
  let res;
  try {
    res = await fetch(`${RL_SERVER}/api/learn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    return NextResponse.json(
      { status: 'error', reason: 'Backend server not reachable. Start it with: cd backend && uvicorn src.server:app --port 8000' },
      { status: 503 }
    );
  }

  if (!res.ok) {
    return NextResponse.json({ status: 'error', message: 'Backend error' }, { status: res.status });
  }

  const data = await res.json();
  return NextResponse.json(data);
}
