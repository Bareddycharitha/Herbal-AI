"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AlertCircle, Leaf, Sparkles } from "lucide-react";
import UploadCard from "@/components/UploadCard";
import { getApiError, predictHerb } from "@/lib/api";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

const guidance = [
  {
    title: "Clear leaf image",
    text: "Capture a high-quality, well-lit photo of the medicinal plant leaf for accurate identification.",
  },
  {
    title: "Secure analysis",
    text: "Your upload is processed securely and not stored beyond the session.",
  },
  {
    title: "Educational resource",
    text: "This tool is designed to support learning about medicinal herbs. Always consult experts before herbal use.",
  },
];

export default function HerbIdentificationPage() {
  const router = useRouter();
  const [error, setError] = useState("");

  const run = async (file: File) => {
    setError("");

    try {
      const result = await predictHerb(file);

      // Handle validation failure
      if (!result.success) {
        setError(result.message || "Image validation failed. Please try another image.");
        return;
      }

      const dataUrl = await fileToDataUrl(file);

      sessionStorage.setItem("herb-prediction", JSON.stringify(result));
      sessionStorage.setItem("herb-image", dataUrl);
      sessionStorage.setItem("herb-file-name", file.name);
      sessionStorage.setItem("herb-file-type", file.type || "image/jpeg");

      router.push("/herb-results");
    } catch (event) {
      const errorObj = getApiError(event);
      setError(errorObj.message);
      throw event;
    }
  };

  return (
    <main className="mx-auto max-w-7xl px-5 py-14 sm:px-8 lg:px-10">
      <div className="mx-auto max-w-3xl text-center">
        <p className="text-sm font-medium uppercase tracking-[0.3em] text-emerald-600">🌿 Herb Identification</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
          Identify Medicinal Plants
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg leading-8 text-muted-foreground">
          Upload a clear image of a medicinal plant leaf to identify the herb, explore its medicinal properties, scientific classification, traditional uses, and supported skin conditions.
        </p>
      </div>

      <div className="mt-10 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <UploadCard onAnalyze={run} workflow="herb" />

        <aside className="space-y-4">
          <div className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.07)]">
            <div className="inline-flex items-center gap-2 rounded-full bg-emerald-100 px-3 py-1 text-sm font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
              <Sparkles size={16} />
              Herb Analysis
            </div>
            <h2 className="mt-5 text-2xl font-semibold">Analysis Features</h2>
            <div className="mt-5 space-y-4">
              {[
                ["🔍", "Herb Identification"],
                ["📚", "Medicinal Properties"],
                ["🌍", "Traditional Uses"],
              ].map(([emoji, label]) => (
                <div key={label} className="flex items-start gap-3 rounded-2xl bg-muted/60 p-3">
                  <span className="mt-0.5 rounded-xl bg-emerald-100 p-2 text-emerald-700 dark:bg-emerald-900/30">
                    {emoji}
                  </span>
                  <div>
                    <p className="font-medium">{label}</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {label === "Herb Identification"
                        ? "AI-powered leaf recognition"
                        : label === "Medicinal Properties"
                        ? "Comprehensive health benefits"
                        : "Historical and cultural uses"}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {guidance.map((item) => (
            <div key={item.title} className="rounded-[1.5rem] border border-border/70 bg-background/70 p-5">
              <h3 className="font-semibold">{item.title}</h3>
              <p className="mt-2 text-sm leading-7 text-muted-foreground">{item.text}</p>
            </div>
          ))}
        </aside>
      </div>

      {error && (
        <div className="mx-auto mt-6 max-w-2xl">
          <Alert variant="destructive">
            <AlertDescription>
              <div className="flex flex-col gap-3">
                <div className="flex items-start gap-2">
                  <AlertCircle size={18} className="mt-0.5 flex-shrink-0" />
                  <p className="text-sm leading-7">{error}</p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setError("")}
                  className="w-fit"
                >
                  Try Another Image
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        </div>
      )}
    </main>
  );
}

function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Unable to read selected image."));
    reader.readAsDataURL(file);
  });
}
