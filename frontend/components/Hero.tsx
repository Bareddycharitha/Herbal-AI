"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, BrainCircuit, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="pointer-events-none absolute left-1/2 top-16 h-[34rem] w-[34rem] -translate-x-1/2 rounded-full bg-primary/15 blur-[130px]" />

      <div className="relative mx-auto grid min-h-[calc(100vh-5rem)] max-w-[1550px] items-center gap-12 px-8 py-16 xl:grid-cols-[1.05fr_.95fr] xl:gap-20 xl:px-12 xl:py-24">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-4 py-2 text-base text-primary">
            <span>🚀</span>
            AI-powered healthcare platform
          </div>

          <h1 className="max-w-[52rem] text-[clamp(3.4rem,5vw,5.8rem)] font-semibold leading-[.98]">
            AI-Powered{" "}
            <span className="text-primary">
              Skin Disease Analysis & Medicinal Herb Intelligence
            </span>
          </h1>

          <p className="mt-8 max-w-2xl text-xl leading-8 text-muted-foreground">
            Detect skin diseases, identify medicinal plants, receive AI-powered
            summaries, herbal recommendations, Grad-CAM visualizations, downloadable
            medical reports, and interact with an intelligent healthcare
            assistant—all in one platform.
          </p>

          <div className="mt-10 flex flex-wrap gap-4">
            <Link href="/diagnose">
              <Button className="h-11 px-5 py-3.5 text-base font-medium">
                Start Analysis
                <ArrowRight size={18} />
              </Button>
            </Link>
            <Link href="/about">
              <Button
                variant="outline"
                className="h-11 px-5 py-3.5 text-base font-medium"
              >
                Learn More
              </Button>
            </Link>
          </div>

          <div className="mt-12 flex gap-7 text-sm text-muted-foreground">
            <span className="flex items-center gap-2">
              <ShieldCheck size={17} className="text-primary" />
              Private uploads
            </span>
            <span className="flex items-center gap-2">
              <BrainCircuit size={17} className="text-primary" />
              AI-assisted
            </span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.15 }}
          className="relative flex items-center justify-center"
        >
          <div className="absolute inset-0 rounded-[2rem] bg-gradient-to-br from-primary/20 via-transparent to-transparent blur-xl" />
          <div className="relative overflow-hidden rounded-[2rem] border border-border/60 bg-muted/30 backdrop-blur-sm">
            <div className="aspect-square max-w-full bg-gradient-to-br from-primary/10 to-transparent p-8">
              <div className="flex h-full w-full items-center justify-center rounded-[1.2rem] border border-primary/20 bg-background/50 backdrop-blur">
                <div className="text-center">
                  <div className="mb-4 text-6xl">🩺 🌿</div>
                  <p className="text-sm font-medium text-muted-foreground">
                    Dual AI Platform
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
