"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { BrainCircuit, Info, ShieldCheck, Sparkles } from "lucide-react";
import UploadCard from "@/components/UploadCard";
import { getApiError, predictImage } from "@/lib/api";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

const guidance = [
  {
    title: "High quality input",
    text: "Use a clear, well-lit image with the affected area centered for a more reliable prediction.",
  },
  {
    title: "Secure analysis",
    text: "Your upload is sent directly to the AI service for analysis and is not stored beyond the session.",
  },
  {
    title: "Clinical caution",
    text: "This experience is designed for educational support and should not replace medical care for urgent concerns.",
  },
];

export default function SkinAnalysisPage() {
  const router = useRouter();
  const [error, setError] = useState("");

  const run = async (file: File) => {
    setError("");

    try {
      const result = await predictImage(file);

      // Handle validation failure
      if (!result.success) {
        setError(result.message || "Image validation failed. Please try another image.");
        return;
      }

      const dataUrl = await fileToDataUrl(file);

      sessionStorage.setItem("diagnosis", JSON.stringify(result));
      sessionStorage.setItem("diagnosis-image", dataUrl);
      sessionStorage.setItem("diagnosis-image-data", dataUrl);
      sessionStorage.setItem("diagnosis-file-name", file.name);
      sessionStorage.setItem("diagnosis-file-type", file.type || "image/jpeg");

      router.push("/results");
    } catch (event) {
      const errorObj = getApiError(event);
      setError(errorObj.message);
      throw event;
    }
  };

  return (
    <main className="mx-auto max-w-7xl px-5 py-14 sm:px-8 lg:px-10">
      <div className="mx-auto max-w-3xl text-center">
        <p className="text-sm font-medium uppercase tracking-[0.3em] text-primary">Skin Disease Analysis</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
          Start your professional skin assessment.
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg leading-8 text-muted-foreground">
          Upload a clear image and the system will return disease prediction, explainability, herbal guidance, and a polished medical summary.
        </p>
      </div>

      <div className="mt-10 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <UploadCard onAnalyze={run} workflow="skin" />

        <aside className="space-y-4">
          <div className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.07)]">
            <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-sm font-medium text-primary">
              <Sparkles size={16} />
              Premium workflow
            </div>
            <h2 className="mt-5 text-2xl font-semibold">What happens next</h2>
            <div className="mt-5 space-y-4">
              {[
                [BrainCircuit, "Prediction"],
                [ShieldCheck, "AI explanation"],
                [Info, "Herbal recommendations"],
              ].map(([Icon, label], index) => {
                const Component = Icon as typeof BrainCircuit;
                return (
                  <div key={label as string} className="flex items-start gap-3 rounded-2xl bg-muted/60 p-3">
                    <span className="mt-0.5 rounded-xl bg-primary/10 p-2 text-primary">
                      <Component size={16} />
                    </span>
                    <div>
                      <p className="font-medium">{label as string}</p>
                      <p className="mt-1 text-sm text-muted-foreground">Step {index + 1} in the guided analysis flow.</p>
                    </div>
                  </div>
                );
              })}
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
                <p className="text-sm leading-7">{error}</p>
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
