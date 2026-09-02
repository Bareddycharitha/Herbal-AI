"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth as useClerkAuth } from "@clerk/nextjs";
import { useAuth } from "@/providers/AuthProvider";

import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import Footer from "@/components/Footer";

export default function Home() {
  const { isAuthenticated } = useAuth();
  // Use Clerk's own ``isLoaded`` to avoid redirecting the user to
  // /login while their session is still hydrating. Without this,
  // there is a brief window where the AuthProvider reports
  // ``isLoading=false`` but Clerk has not yet told us whether the
  // user is signed in, and we would incorrectly bounce a signed-in
  // user to the login page.
  const { isLoaded: clerkLoaded } = useClerkAuth();
  const router = useRouter();

  useEffect(() => {
    if (clerkLoaded && !isAuthenticated) {
      router.replace("/login");
    }
  }, [isAuthenticated, clerkLoaded, router]);

  // Show a spinner while Clerk is still hydrating. Once Clerk knows
  // whether the user is signed in, we either render the homepage
  // (signed in) or the redirect above kicks in (signed out).
  if (!clerkLoaded) {
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