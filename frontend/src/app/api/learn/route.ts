import { NextRequest, NextResponse } from 'next/server';

const RL_SERVER = process.env.RL_SERVER || 'http://localhost:8000';

export async function POST(req: NextRequest) {
  const body = await req.json();
  const res = await fetch(`${RL_SERVER}/api/learn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    return NextResponse.json({ status: 'error', message: 'Backend error' }, { status: res.status });
  }

  const data = await res.json();
  return NextResponse.json(data);
}
