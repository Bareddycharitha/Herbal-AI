"use client";

import Link from "next/link";
import { Heart, Leaf, Mail } from "lucide-react";

const quickLinks = [
  {
    title: "Home",
    href: "/",
  },
  {
    title: "Diagnose",
    href: "/diagnose",
  },
  {
    title: "About",
    href: "/about",
  },
];

export default function Footer() {
  return (
    <footer className="border-t border-border bg-background">
      <div className="mx-auto max-w-7xl px-6 py-14">
        <div className="grid gap-12 md:grid-cols-3">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-emerald-500 p-2 text-white">
                <Leaf size={18} />
              </div>

              <h2 className="text-2xl font-bold">
                Herbal-AI
              </h2>
            </div>

            <p className="mt-5 max-w-sm leading-7 text-muted-foreground">
              AI-powered skin disease detection with intelligent herbal
              recommendations, designed to assist users in understanding
              potential skin conditions quickly and efficiently.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="mb-5 text-lg font-semibold">
              Quick Links
            </h3>

            <div className="flex flex-col gap-3">
              {quickLinks.map((link) => (
                <Link
                  key={link.title}
                  href={link.href}
                  className="text-muted-foreground transition-colors hover:text-emerald-500"
                >
                  {link.title}
                </Link>
              ))}
            </div>
          </div>

          {/* Contact */}
          <div>
            <h3 className="mb-5 text-lg font-semibold">
              Connect
            </h3>

            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-3 text-muted-foreground">
                <Mail size={18} />
                <span>support@herbal-ai.com</span>
              </div>

              <div className="flex gap-4 pt-2">
                <Link
                  href="#"
                  className="rounded-lg border border-border p-2 transition-all hover:border-emerald-500 hover:text-emerald-500"
                >
                 <span className="text-sm font-medium">GitHub</span>
                </Link>

                <Link
                  href="#"
                  className="rounded-lg border border-border p-2 transition-all hover:border-emerald-500 hover:text-emerald-500"
                >
                  <span className="text-sm font-medium">LinkedIn</span>
                </Link>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-border pt-8 text-sm text-muted-foreground md:flex-row">
          <p>
            © {new Date().getFullYear()} Herbal-AI. All rights reserved.
          </p>

          <p className="flex items-center gap-2">
            Built with
            <Heart className="h-4 w-4 fill-red-500 text-red-500" />
            using Next.js & AI
          </p>
        </div>
      </div>
    </footer>
  );
}