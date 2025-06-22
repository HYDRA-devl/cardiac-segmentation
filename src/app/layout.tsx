import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Pipeline Échographique IA',
  description: 'Interface professionnelle pour l\'amélioration d\'images échographiques par intelligence artificielle',
  keywords: [
    'échographie',
    'intelligence artificielle',
    'traitement d\'image',
    'segmentation',
    'débruitage',
    'super-résolution'
  ],
  authors: [{ name: 'Équipe Pipeline IA' }],
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="fr" className="h-full">
      <body className={`${inter.className} h-full overflow-hidden`}>
        <div className="h-full">
          {children}
        </div>
      </body>
    </html>
  );
}
