"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  Leaf,
  Menu,
  X,
  User,
  LogOut,
  LogIn,
  UserPlus,
  Sparkles,
  History,
} from "lucide-react";
import ThemeToggle from "./ThemeToggle";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/providers/AuthProvider";

const baseLinks = [
  ["Home", "/"],
  ["Skin Analysis", "/skin-analysis"],
  ["Herb Identification", "/herb-identification"],
  ["AI Chat", "/chat"],
  ["About", "/about"],
];

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const { isAuthenticated, user, logout } = useAuth();

  const navLinks = isAuthenticated
    ? [
        ["Home", "/"],
        ["Skin Analysis", "/skin-analysis"],
        ["Herb Identification", "/herb-identification"],
        ["AI Chat", "/chat"],
        ["History", "/history"],
        ["About", "/about"],
      ]
    : baseLinks;

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/80 backdrop-blur-xl transition-colors">
      <div className="mx-auto flex h-[5.25rem] max-w-[1500px] items-center justify-between px-6 sm:px-8">
        {/* Brand Logo */}
        <Link href="/" className="flex items-center gap-3 text-xl font-bold tracking-tight text-foreground">
          <span className="grid h-11 w-11 place-items-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
            <Leaf size={22} />
          </span>
          <span>Herbal-AI</span>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="hidden items-center gap-8 text-[15px] font-medium md:flex lg:gap-10">
          {navLinks.map(([n, h]) => {
            const active = isActive(h);
            return (
              <Link
                key={n}
                href={h}
                className={`relative py-1 transition-all ${
                  active
                    ? "font-semibold text-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {n}
                {active && (
                  <span className="absolute inset-x-0 -bottom-1.5 h-0.5 rounded-full bg-primary" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Desktop User Controls */}
        <div className="hidden items-center gap-3.5 md:flex">
          <ThemeToggle />

          {isAuthenticated ? (
            <div className="flex items-center gap-3">
              <span className="max-w-[180px] truncate text-sm font-medium text-muted-foreground" title={user?.full_name || user?.email}>
                {user?.full_name || user?.email}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={logout}
                className="h-10 rounded-xl px-4 text-xs font-semibold"
              >
                <LogOut className="mr-2 h-3.5 w-3.5" />
                Logout
              </Button>
            </div>
          ) : (
            <div className="flex items-center gap-2.5">
              <Link href="/login">
                <Button variant="ghost" size="sm" className="h-10 rounded-xl px-4 text-xs font-semibold">
                  <LogIn className="mr-1.5 h-3.5 w-3.5" />
                  Sign In
                </Button>
              </Link>
              <Link href="/register">
                <Button variant="outline" size="sm" className="h-10 rounded-xl px-4 text-xs font-semibold">
                  <UserPlus className="mr-1.5 h-3.5 w-3.5" />
                  Sign Up
                </Button>
              </Link>
            </div>
          )}

          <Link href="/diagnose">
            <Button className="h-10 rounded-xl px-5 text-xs font-semibold shadow-sm">
              <Sparkles className="mr-1.5 h-3.5 w-3.5" />
              Start Analysis
            </Button>
          </Link>
        </div>

        {/* Mobile Hamburger Button */}
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

      {/* Mobile Drawer Menu */}
      {open && (
        <div className="border-t border-border bg-background px-6 py-6 shadow-xl md:hidden">
          <div className="flex flex-col gap-4">
            {navLinks.map(([n, h]) => {
              const active = isActive(h);
              return (
                <Link
                  onClick={() => setOpen(false)}
                  key={n}
                  href={h}
                  className={`rounded-xl px-3 py-2 text-base transition-colors ${
                    active
                      ? "bg-primary/10 font-semibold text-primary"
                      : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                  }`}
                >
                  {n}
                </Link>
              );
            })}

            <div className="my-2 border-t border-border/60" />

            {isAuthenticated ? (
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-3 px-3 py-1 text-sm text-muted-foreground">
                  <User size={16} />
                  <span className="truncate font-medium">{user?.full_name || user?.email}</span>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setOpen(false);
                    logout();
                  }}
                  className="w-full justify-start rounded-xl text-destructive hover:text-destructive"
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Logout
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                <Link href="/login" onClick={() => setOpen(false)}>
                  <Button variant="outline" className="w-full rounded-xl">
                    <LogIn className="mr-2 h-4 w-4" />
                    Sign In
                  </Button>
                </Link>
                <Link href="/register" onClick={() => setOpen(false)}>
                  <Button className="w-full rounded-xl">
                    <UserPlus className="mr-2 h-4 w-4" />
                    Sign Up
                  </Button>
                </Link>
              </div>
            )}

            <div className="mt-2 flex items-center justify-between pt-2">
              <ThemeToggle />
              <Link href="/diagnose" onClick={() => setOpen(false)}>
                <Button className="rounded-xl">Start Analysis</Button>
              </Link>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
