"use client";

import { useAuth } from "@/providers/AuthProvider";

import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import Footer from "@/components/Footer";

export default function Home() {
  const { isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-muted-foreground">
          <div className="size-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          <p className="text-sm">Loading…</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Navbar />

      <main>
        <Hero />
        <Features />
      </main>

      <Footer />
    </>
  );
}