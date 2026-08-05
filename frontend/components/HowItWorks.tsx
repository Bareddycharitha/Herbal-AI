"use client";

import { motion } from "framer-motion";
import {
  Upload,
  Brain,
  Activity,
  Leaf,
  ArrowDown,
  ArrowRight,
} from "lucide-react";

const steps = [
  {
    icon: Upload,
    title: "Upload Image",
    description:
      "Upload a clear image of the affected skin area for AI analysis.",
  },
  {
    icon: Brain,
    title: "AI Analysis",
    description:
      "Our deep learning model analyzes patterns, textures, and skin characteristics.",
  },
  {
    icon: Activity,
    title: "Disease Prediction",
    description:
      "The system predicts the most likely skin disease with a confidence score.",
  },
  {
    icon: Leaf,
    title: "Herbal Recommendation",
    description:
      "Receive herbal remedies, precautions, and treatment suggestions.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24">
      <div className="mx-auto max-w-7xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 25 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mx-auto max-w-3xl text-center"
        >
          <h2 className="text-4xl font-bold tracking-tight">
            How Herbal-AI Works
          </h2>

          <p className="mt-4 text-lg text-muted-foreground">
            A simple four-step workflow that transforms a skin image into
            meaningful AI insights and herbal recommendations.
          </p>
        </motion.div>

        <div className="mt-20 grid gap-10 md:grid-cols-2 lg:grid-cols-4">
          {steps.map((step, index) => {
            const Icon = step.icon;

            return (
              <div key={step.title} className="relative">
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{
                    delay: index * 0.12,
                  }}
                  className="rounded-2xl border border-border bg-background/60 p-7 text-center backdrop-blur"
                >
                  <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-500/10 text-emerald-500">
                    <Icon size={30} />
                  </div>

                  <div className="mb-3 text-sm font-semibold text-emerald-500">
                    STEP {index + 1}
                  </div>

                  <h3 className="text-xl font-semibold">
                    {step.title}
                  </h3>

                  <p className="mt-3 leading-7 text-muted-foreground">
                    {step.description}
                  </p>
                </motion.div>

                {/* Desktop Connector */}
                {index < steps.length - 1 && (
                  <div className="absolute right-[-24px] top-1/2 hidden -translate-y-1/2 lg:block">
                    <ArrowRight className="text-emerald-500" size={22} />
                  </div>
                )}

                {/* Tablet Connector */}
                {index < steps.length - 1 && (
                  <div className="mt-6 flex justify-center lg:hidden">
                    <ArrowDown className="text-emerald-500" size={22} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
