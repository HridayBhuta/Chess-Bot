import type { Metadata } from 'next';
import Script from 'next/script';
import './globals.css';

export const metadata: Metadata = {
  title: 'Chess Bot',
  description: 'Play chess against an AI trained on your style',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Script src="/ort.wasm.min.js" strategy="beforeInteractive" />
        {children}
      </body>
    </html>
  );
}
