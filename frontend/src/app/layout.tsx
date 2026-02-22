import type { Metadata } from "next";
import { JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

const jetbrains = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-display",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Trading Deportivo",
  description: "Sistema de trading deportivo con modelo Dixon-Coles",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" className="dark">
      <body
        className={`${jetbrains.variable} ${spaceGrotesk.variable} antialiased bg-[#0a0e17] text-slate-200 font-mono min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}
