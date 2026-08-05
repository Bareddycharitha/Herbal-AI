"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Download,
  Leaf,
  Sparkles,
  X,
  ChevronDown,
  Check,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { HerbPredictionResponse } from "@/types";
import { toast } from "sonner";
import UploadCard from "@/components/UploadCard";
import { getApiError, predictHerb } from "@/lib/api";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function HerbResultsPage() {
  const [data, setData] = useState<HerbPredictionResponse | null>(null);
  const [image, setImage] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    medicinal_properties: true,
    traditional_uses: true,
    preparation_methods: true,
    skin_conditions: true,
    precautions: true,
    active_compounds: true,
  });

  useEffect(() => {
    try {
      const storedData = sessionStorage.getItem("herb-prediction");
      const storedImage = sessionStorage.getItem("herb-image") ?? "";
      const storedName = sessionStorage.getItem("herb-file-name") ?? "upload.jpg";
      const storedType = sessionStorage.getItem("herb-file-type") ?? "image/jpeg";

      if (storedData) {
        const parsed = JSON.parse(storedData) as HerbPredictionResponse;
        setData(parsed);
        setImage(storedImage);

        if (storedImage) {
          setFile(dataUrlToFile(storedImage, storedName, storedType));
        }
      }
    } catch {
      toast.error("Could not load previous herb analysis.");
    }
  }, []);

  const handleAnalyzeAnother = async (file: File) => {
    try {
      const result = await predictHerb(file);

      if (!result.success) {
        toast.error(result.message || "Image validation failed");
        return;
      }

      const dataUrl = await fileToDataUrl(file);

      setData(result);
      setImage(dataUrl);
      setFile(dataUrlToFile(dataUrl, file.name, file.type || "image/jpeg"));

      sessionStorage.setItem("herb-prediction", JSON.stringify(result));
      sessionStorage.setItem("herb-image", dataUrl);
      sessionStorage.setItem("herb-file-name", file.name);
      sessionStorage.setItem("herb-file-type", file.type || "image/jpeg");

      toast.success("New herb image analyzed");
    } catch (event) {
      const errorObj = getApiError(event);
      toast.error(errorObj.message);
    }
  };

  if (!data) {
    return (
      <main className="mx-auto flex min-h-screen max-w-2xl items-center px-5 py-24 text-center">
        <div className="w-full rounded-[2rem] border border-border/70 bg-card/85 p-10 shadow-[0_20px_70px_rgba(0,0,0,0.08)]">
          <Sparkles className="mx-auto text-primary" size={28} />
          <h1 className="mt-5 text-3xl font-semibold">No analysis has been started yet</h1>
          <p className="mt-3 text-lg leading-8 text-muted-foreground">
            Upload a medicinal plant leaf image to unlock herb identification and traditional knowledge.
          </p>
          <Link
            href="/herb-identification"
            className="mt-8 inline-flex items-center gap-2 rounded-full bg-primary px-5 py-3 text-sm font-medium text-primary-foreground"
          >
            Start Herb Identification
          </Link>
        </div>
      </main>
    );
  }

  const herbInfo = data.herb_information;
  const confidence = Number(data.prediction?.confidence) || 0;
  const confidenceLevel = data.prediction?.confidence_level?.toLowerCase() || "medium";
  const predictions = data.top_predictions ?? [];

  return (
    <main className="mx-auto max-w-7xl px-5 py-10 sm:px-8 lg:px-10">
      {lightboxImage && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4"
          onClick={() => setLightboxImage(null)}
        >
          <button
            onClick={() => setLightboxImage(null)}
            className="absolute right-4 top-4 rounded-full bg-white/90 p-2 text-foreground"
            aria-label="Close preview"
          >
            <X size={18} />
          </button>
          <img
            src={lightboxImage}
            alt="Expanded preview"
            className="max-h-[90vh] max-w-full rounded-2xl object-contain"
          />
        </div>
      )}

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium uppercase tracking-[0.3em] text-emerald-600">Herb Analysis Report</p>
          <h1 className="mt-2 text-4xl font-semibold tracking-tight sm:text-5xl">
            {herbInfo?.common_name || "Herb Identified"}
          </h1>
          <p className="mt-3 max-w-2xl text-lg leading-8 text-muted-foreground">
            Comprehensive identification, medicinal properties, traditional uses, and botanical information.
          </p>
        </div>
      </div>

      <div className="mt-8 grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
        <div className="space-y-6">
          {/* Main Herb Card */}
          <motion.article
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-[2rem] border border-emerald-200 bg-gradient-to-br from-emerald-50 to-teal-50 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.08)] dark:border-emerald-900/50 dark:from-emerald-950/50 dark:to-teal-950/50"
          >
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <div className="inline-flex items-center gap-2 rounded-full bg-emerald-100 px-3 py-1 text-sm font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                  <Check size={16} />
                  Herb Identified
                </div>
                <h2 className="mt-4 text-3xl font-semibold text-emerald-900 dark:text-emerald-100">
                  {herbInfo?.common_name || "Medicinal Plant"}
                </h2>
                <p className="mt-2 text-sm text-emerald-700 dark:text-emerald-300">
                  {herbInfo?.scientific_name || "Scientific name unavailable"}
                </p>
                {herbInfo?.family && (
                  <p className="mt-1 text-sm text-emerald-600 dark:text-emerald-400">
                    Family: {herbInfo.family}
                  </p>
                )}
              </div>
              <div
                className={`rounded-2xl border px-4 py-3 text-right ${
                  confidenceLevel === "high"
                    ? "border-green-200 bg-green-100 dark:border-green-900/50 dark:bg-green-900/30"
                    : confidenceLevel === "medium"
                    ? "border-amber-200 bg-amber-100 dark:border-amber-900/50 dark:bg-amber-900/30"
                    : "border-red-200 bg-red-100 dark:border-red-900/50 dark:bg-red-900/30"
                }`}
              >
                <p
                  className={`text-sm ${
                    confidenceLevel === "high"
                      ? "text-green-700 dark:text-green-400"
                      : confidenceLevel === "medium"
                      ? "text-amber-700 dark:text-amber-400"
                      : "text-red-700 dark:text-red-400"
                  }`}
                >
                  {confidenceLevel.charAt(0).toUpperCase() + confidenceLevel.slice(1)} Confidence
                </p>
                <p
                  className={`mt-1 text-3xl font-semibold ${
                    confidenceLevel === "high"
                      ? "text-green-600 dark:text-green-400"
                      : confidenceLevel === "medium"
                      ? "text-amber-600 dark:text-amber-400"
                      : "text-red-600 dark:text-red-400"
                  }`}
                >
                  {confidence.toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="mt-6 h-2 overflow-hidden rounded-full bg-emerald-200 dark:bg-emerald-900/50">
              <div
                className={`h-full rounded-full ${
                  confidenceLevel === "high"
                    ? "bg-green-500"
                    : confidenceLevel === "medium"
                    ? "bg-amber-500"
                    : "bg-red-500"
                }`}
                style={{ width: `${Math.min(confidence, 100)}%` }}
              />
            </div>

            {herbInfo?.description && (
              <p className="mt-6 text-sm leading-7 text-emerald-800 dark:text-emerald-200">
                {herbInfo.description}
              </p>
            )}
          </motion.article>

          {/* Image */}
          {image && (
            <motion.article
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="rounded-[2rem] border border-border/70 bg-card/85 p-5 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
            >
              <h3 className="text-xl font-semibold">Uploaded Image</h3>
              <div className="mt-4 overflow-hidden rounded-[1.5rem] border border-border/70 bg-muted/40">
                <img
                  src={image}
                  alt="Uploaded herb image"
                  className="h-72 w-full cursor-zoom-in object-cover"
                  onClick={() => setLightboxImage(image)}
                />
                <div className="p-3 text-sm text-muted-foreground">Submitted leaf image</div>
              </div>
            </motion.article>
          )}

          {/* Analyze Another */}
          <motion.article
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
          >
            <h3 className="text-xl font-semibold mb-4">Analyze Another Herb</h3>
            <UploadCard onAnalyze={handleAnalyzeAnother} workflow="herb" compact />
          </motion.article>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Top Predictions */}
          {predictions.length > 0 && (
            <motion.article
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.12 }}
              className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
            >
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-primary/10 p-2 text-primary">
                  <Sparkles size={20} />
                </div>
                <div>
                  <h3 className="text-xl font-semibold">Top Predictions</h3>
                  <p className="text-sm text-muted-foreground">The model's most likely alternatives.</p>
                </div>
              </div>
              <div className="mt-5 space-y-4">
                {predictions.map((item) => (
                  <div key={`${item.herb}-${item.confidence}`}>
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="font-medium">{item.herb}</span>
                      <span className="text-muted-foreground">{item.confidence.toFixed(1)}%</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-emerald-500"
                        style={{ width: `${Math.min(item.confidence, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </motion.article>
          )}

          {/* Medicinal Properties */}
          {herbInfo?.medicinal_properties && herbInfo.medicinal_properties.length > 0 && (
            <ExpandableSection
              title="Medicinal Properties"
              delay={0.14}
              items={herbInfo.medicinal_properties}
              icon="💊"
              expandedSections={expandedSections}
              setExpandedSections={setExpandedSections}
              id="medicinal_properties"
            />
          )}

          {/* Traditional Uses */}
          {herbInfo?.traditional_uses && herbInfo.traditional_uses.length > 0 && (
            <ExpandableSection
              title="Traditional Uses"
              delay={0.16}
              items={herbInfo.traditional_uses}
              icon="📖"
              expandedSections={expandedSections}
              setExpandedSections={setExpandedSections}
              id="traditional_uses"
            />
          )}

          {/* Preparation Methods */}
          {herbInfo?.preparation_methods && herbInfo.preparation_methods.length > 0 && (
            <ExpandableSection
              title="Preparation Methods"
              delay={0.18}
              items={herbInfo.preparation_methods}
              icon="🍵"
              expandedSections={expandedSections}
              setExpandedSections={setExpandedSections}
              id="preparation_methods"
            />
          )}

          {/* Skin Conditions Supported */}
          {herbInfo?.skin_conditions_supported && herbInfo.skin_conditions_supported.length > 0 && (
            <ExpandableSection
              title="Skin Conditions Supported"
              delay={0.2}
              items={herbInfo.skin_conditions_supported}
              icon="🩺"
              expandedSections={expandedSections}
              setExpandedSections={setExpandedSections}
              id="skin_conditions"
            />
          )}

          {/* Active Compounds */}
          {herbInfo?.active_compounds && herbInfo.active_compounds.length > 0 && (
            <ExpandableSection
              title="Active Compounds"
              delay={0.22}
              items={herbInfo.active_compounds}
              icon="⚗️"
              expandedSections={expandedSections}
              setExpandedSections={setExpandedSections}
              id="active_compounds"
            />
          )}

          {/* Precautions */}
          {herbInfo?.precautions && herbInfo.precautions.length > 0 && (
            <motion.article
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.24 }}
              className="rounded-[2rem] border border-amber-200/50 bg-amber-50/50 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)] dark:border-amber-900/30 dark:bg-amber-900/20"
            >
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-amber-100/80 p-2 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                  <Leaf size={20} />
                </div>
                <div>
                  <h3 className="text-xl font-semibold">Precautions</h3>
                  <p className="text-sm text-amber-700 dark:text-amber-300">
                    Important safety information
                  </p>
                </div>
              </div>
              <ul className="mt-4 space-y-2">
                {herbInfo.precautions.map((precaution, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                    <span className="text-amber-600 dark:text-amber-400">⚠️</span>
                    <span>{precaution}</span>
                  </li>
                ))}
              </ul>
            </motion.article>
          )}
        </div>
      </div>
    </main>
  );
}

function ExpandableSection({
  title,
  delay,
  items,
  icon,
  expandedSections,
  setExpandedSections,
  id,
}: {
  title: string;
  delay: number;
  items: string[];
  icon: string;
  expandedSections: Record<string, boolean>;
  setExpandedSections: (sections: Record<string, boolean>) => void;
  id: string;
}) {
  const isExpanded = expandedSections[id];

  return (
    <motion.article
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
    >
      <button
        onClick={() =>
          setExpandedSections({
            ...expandedSections,
            [id]: !isExpanded,
          })
        }
        className="flex w-full items-center justify-between"
      >
        <div className="flex items-center gap-3">
          <span className="text-xl">{icon}</span>
          <h3 className="text-lg font-semibold">{title}</h3>
        </div>
        <ChevronDown
          size={20}
          className={`transition-transform ${isExpanded ? "rotate-180" : ""}`}
        />
      </button>

      {isExpanded && (
        <ul className="mt-4 space-y-2">
          {items.map((item, idx) => (
            <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
              <span className="mt-1 text-emerald-500 flex-shrink-0">•</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )}
    </motion.article>
  );
}

function dataUrlToFile(dataUrl: string, fileName: string, type: string) {
  if (!dataUrl) {
    return new File([""], fileName, { type });
  }

  const byteString = atob(dataUrl.split(",")[1] ?? "");
  const arrayBuffer = new Uint8Array(byteString.length);
  for (let index = 0; index < byteString.length; index += 1) {
    arrayBuffer[index] = byteString.charCodeAt(index);
  }
  return new File([arrayBuffer], fileName, { type });
}

async function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Unable to read selected image."));
    reader.readAsDataURL(file);
  });
}
