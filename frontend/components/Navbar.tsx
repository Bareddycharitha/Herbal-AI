"use client";
import Link from "next/link";
import { useState } from "react";
import { Leaf, Menu, X, User, LogOut, LogIn, UserPlus } from "lucide-react";
import ThemeToggle from "./ThemeToggle";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/providers/AuthProvider";

const links = [
  ["Home", "/"],
  ["Skin Analysis", "/skin-analysis"],
  ["Herb Identification", "/herb-identification"],
  ["AI Chat", "/chat"],
  ["About", "/about"],
];

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const { isAuthenticated, user, logout } = useAuth();

  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/75 backdrop-blur-xl">
      <div className="mx-auto flex h-[5.5rem] max-w-[1500px] items-center justify-between px-8">
        <Link href="/" className="flex items-center gap-3.5 text-xl font-semibold">
          <span className="grid h-12 w-12 place-items-center rounded-2xl bg-primary text-primary-foreground">
            <Leaf size={23} />
          </span>
          Herbal-AI
        </Link>

        <nav className="hidden gap-11 text-base text-muted-foreground md:flex">
          {links.map(([n, h]) => (
            <Link
              key={n}
              href={h}
              className="transition-colors hover:text-foreground"
            >
              {n}
            </Link>
          ))}
        </nav>

        <div className="hidden items-center gap-4 md:flex">
          <ThemeToggle />
          {isAuthenticated ? (
            <>
              <span className="text-sm text-muted-foreground">
                {user?.full_name || user?.email}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={logout}
                className="h-11 px-5"
              >
                <LogOut className="mr-2 h-4 w-4" />
                Logout
              </Button>
            </>
          ) : (
            <>
              <Link href="/login">
                <Button variant="outline" className="h-11 px-5">
                  <LogIn className="mr-2 h-4 w-4" />
                  Sign in
                </Button>
              </Link>
              <Link href="/register">
                <Button className="h-11 px-5">
                  <UserPlus className="mr-2 h-4 w-4" />
                  Sign up
                </Button>
              </Link>
            </>
          )}
          <Link href="/diagnose">
            <Button className="h-11 px-5 text-[15px]">Start Analysis</Button>
          </Link>
        </div>

        <Button
          aria-label="Toggle menu"
          variant="ghost"
          size="icon-lg"
          className="md:hidden"
          onClick={() => setOpen(!open)}
        >
          {open ? <X /> : <Menu />}
        </Button>
      </div>

      {open && (
        <div className="border-t border-border bg-background px-8 py-5 md:hidden">
          <div className="flex flex-col gap-5">
            {links.map(([n, h]) => (
              <Link
                onClick={() => setOpen(false)}
                key={n}
                href={h}
              >
                {n}
              </Link>
            ))}
            {isAuthenticated && (
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <User size={16} />
                <span>{user?.full_name || user?.email}</span>
              </div>
            )}
            <div className="flex justify-between">
              <ThemeToggle />
              <Link href="/diagnose">
                <Button>Start Analysis</Button>
              </Link>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
