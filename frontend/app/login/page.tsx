"use client";

import { SignIn } from "@clerk/nextjs";

export default function LoginPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-md items-center px-5 py-16">
      <div className="w-full rounded-[2rem] border border-border/70 bg-card/85 p-8 shadow-[0_24px_70px_rgba(0,0,0,.07)] backdrop-blur sm:p-10">
        <div className="text-center">
          <h1 className="text-3xl font-semibold">Welcome back</h1>
          <p className="mt-3 text-base text-muted-foreground">
            Sign in to access skin disease analysis and herb identification.
          </p>
        </div>

        <div className="mt-8">
          <SignIn />
        </div>
      </div>
    </main>
  );
}
