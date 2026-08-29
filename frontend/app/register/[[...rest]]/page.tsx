"use client";

import { SignUp } from "@clerk/nextjs";

export default function RegisterPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-5 py-10">
      <SignUp
        path="/register"
        routing="path"
        signInUrl="/login"
        forceRedirectUrl="/"
      />
    </main>
  );
}