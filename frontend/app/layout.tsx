import type { Metadata } from "next";
import "./globals.css";

import { Toaster } from "sonner";
import ThemeProvider from "@/providers/ThemeProvider";
import { AuthProvider } from "@/providers/AuthProvider";

export const metadata: Metadata = {
  title: "Herbal-AI",
  description:
    "AI-powered skin disease diagnosis and herbal treatment recommendations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-scroll-behavior="smooth" suppressHydrationWarning>
      <body>
        <AuthProvider>
          <ThemeProvider>{children}</ThemeProvider>
        </AuthProvider>
        <Toaster />
      </body>
    </html>
  );
}
