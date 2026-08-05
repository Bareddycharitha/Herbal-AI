"use client";
import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { AlertCircle, LoaderCircle, RefreshCw, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";

type Props = {
  onAnalyze: (file: File) => Promise<void>;
  compact?: boolean;
  workflow?: "skin" | "herb";
};

const MAX_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg"];

export default function UploadCard({ onAnalyze, compact = false, workflow = "skin" }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const steps = workflow === "skin"
    ? ["Validating image", "Checking medical image", "Detecting disease", "Retrieving knowledge base", "Generating AI summary", "Preparing report"]
    : ["Extracting leaf features", "Running AI model", "Matching knowledge base", "Preparing herb information"];

  const select = (candidate: File) => {
    setError("");

    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setError("Please choose a JPG, PNG, or JPEG image.");
      return;
    }

    if (candidate.size > MAX_SIZE) {
      setError("Images must be smaller than 10 MB.");
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(candidate);
    setPreview(URL.createObjectURL(candidate));
  };

  const reset = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setFile(null);
    setPreview(null);
    setError("");
    if (input.current) {
      input.current.value = "";
    }
  };

  const analyse = async () => {
    if (!file) return;
    setLoading(true);
    try {
      await onAnalyze(file);
    } catch {
      setLoading(false);
    }
  };

  return (
    <section className="rounded-[2rem] border border-border/70 bg-card/85 p-4 shadow-[0_30px_80px_rgba(0,0,0,0.08)] backdrop-blur sm:p-7">
      {preview ? (
        <div className="space-y-4">
          <div className="relative overflow-hidden rounded-[1.5rem] bg-muted">
            <img
              src={preview}
              alt="Selected skin image"
              className={`w-full object-cover ${compact ? "h-60" : "h-72"}`}
            />
            <button
              aria-label="Remove image"
              onClick={reset}
              className="absolute right-3 top-3 rounded-full bg-background/85 p-2"
            >
              <X size={17} />
            </button>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row">
            <Button variant="outline" onClick={() => input.current?.click()} className="flex-1">
              <RefreshCw />
              Replace image
            </Button>
            <Button disabled={loading} onClick={analyse} className="flex-1">
              {loading ? (
                <>
                  <LoaderCircle className="animate-spin" />
                  Analyzing image…
                </>
              ) : (
                <>{workflow === "skin" ? "Analyze image" : "Identify herb"}</>
              )}
            </Button>
          </div>
          {loading && (
            <div className="grid gap-1.5 rounded-2xl bg-primary/5 p-4 text-sm text-muted-foreground sm:grid-cols-2">
              {steps.map((step, index) => <p key={step} className="animate-pulse" style={{ animationDelay: `${index * 120}ms` }}>✓ {step}</p>)}
            </div>
          )}
        </div>
      ) : (
        <div
          onClick={() => input.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event: DragEvent) => {
            event.preventDefault();
            setDragging(false);
            const picked = event.dataTransfer.files[0];
            if (picked) {
              select(picked);
            }
          }}
          className={`cursor-pointer rounded-[1.6rem] border border-dashed p-8 text-center transition sm:p-12 ${dragging ? "border-primary bg-primary/10" : "border-border/80 hover:border-primary hover:bg-primary/5"}`}
        >
          <span className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-2xl bg-primary/10 text-primary">
            <Upload size={28} />
          </span>
          <h2 className="text-2xl font-semibold">{workflow === "skin" ? "Upload a skin image" : "Upload a medicinal leaf image"}</h2>
          <p className="mt-3 text-base text-muted-foreground">
            Drag and drop or <span className="font-medium text-primary">browse files</span>
          </p>
          <p className="mt-6 text-sm text-muted-foreground">PNG, JPG, JPEG · Maximum 10 MB</p>
        </div>
      )}

      {error && (
        <p role="alert" className="mt-4 flex items-center gap-2 text-sm text-destructive">
          <AlertCircle size={16} />
          {error}
        </p>
      )}

      <input
        ref={input}
        className="hidden"
        type="file"
        accept="image/png,image/jpeg"
        onChange={(event: ChangeEvent<HTMLInputElement>) => {
          const picked = event.target.files?.[0];
          if (picked) {
            select(picked);
          }
        }}
      />
    </section>
  );
}
