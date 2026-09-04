"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Bot,
  Check,
  Download,
  FileText,
  Leaf,
  LoaderCircle,
  MessageCircleMore,
  Sparkles,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  chatWithAI,
  downloadReport,
  fetchGradcamStatus,
  generateSummary,
  getApiBaseUrl,
  getApiError,
  predictImage,
} from "@/lib/api";
import type { Herb, PredictionResponse } from "@/types";
import { toast } from "sonner";
import UploadCard from "@/components/UploadCard";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useAuth } from "@/providers/AuthProvider";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

const suggestions = [
  "Explain this disease",
  "What precautions should I take?",
  "Are the recommended herbs safe?",
  "How long does recovery usually take?",
  "Can this condition spread?",
];

const healthyTips = [
  "Moisturize regularly",
  "Wear sunscreen daily",
  "Stay well hydrated",
  "Maintain a healthy diet",
  "Consult a dermatologist if changes occur",
];

export default function ResultsPage() {
  const { isAuthenticated, tokenReady } = useAuth();
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [image, setImage] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [summary, setSummary] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);

  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  useEffect(() => {
    try {
      const storedData = sessionStorage.getItem("diagnosis");
      const storedImage = sessionStorage.getItem("diagnosis-image") ?? "";
      const storedName = sessionStorage.getItem("diagnosis-file-name") ?? "upload.jpg";
      const storedType = sessionStorage.getItem("diagnosis-file-type") ?? "image/jpeg";

      if (storedData) {
        const parsed = JSON.parse(storedData) as PredictionResponse;
        setData(parsed);
        setImage(storedImage);

        if (storedImage) {
          setFile(dataUrlToFile(storedImage, storedName, storedType));
        }
      }
    } catch {
      setSummaryError("Your previous diagnosis data could not be loaded.");
    }
  }, []);

  // Sync the Grad-CAM URL state from the prediction response, then poll
  // the status endpoint when the response came back with gradcam_image=null
  // (which happens on the new fast path where the heatmap is generated in
  // the background after the response is sent).
  useEffect(() => {
    if (!data) {
      setGradcamUrl(null);
      return;
    }

    const initial = data.gradcam_image;
    if (initial) {
      const absolute = initial.startsWith("http")
        ? initial
        : `${apiBaseUrl}${initial}`;
      setGradcamUrl(absolute);
      return;
    }

    // No URL yet — if we have a prediction_id, poll for it.
    const pid = data.prediction_id;
    if (!pid) {
      setGradcamUrl(null);
      return;
    }

    let cancelled = false;
    let attempts = 0;
    const maxAttempts = 20;
    const intervalMs = 1500;

    const tick = async () => {
      if (cancelled || attempts >= maxAttempts) return;
      attempts += 1;
      try {
        const status = await fetchGradcamStatus(pid);
        if (cancelled) return;
        if (status.ready && status.url) {
          const absolute = status.url.startsWith("http")
            ? status.url
            : `${apiBaseUrl}${status.url}`;
          setGradcamUrl(absolute);
          return;
        }
        if (status.status === "no_gradcam" || status.status === "failed") {
          // Terminal state without a URL — leave gradcamUrl null so the
          // UI shows the single-image "Uploaded image" view.
          setGradcamUrl(null);
          return;
        }
      } catch {
        // Network blip — try again until maxAttempts.
      }
      if (!cancelled && attempts < maxAttempts) {
        setTimeout(tick, intervalMs);
      }
    };

    setGradcamUrl(null);
    setTimeout(tick, 0);

    return () => {
      cancelled = true;
    };
  }, [data, apiBaseUrl]);

  const loadSummary = async () => {
    if (!data) return;

    setSummaryLoading(true);
    setSummaryError("");

    try {
      const response = await generateSummary({
        prediction: data.prediction!.disease,
        confidence: data.prediction!.confidence,
        disease_information: data.disease_information ?? {},
        herbs: data.recommended_herbs ?? [],
      });
      if (response.success) {
        setSummary(response.summary || "");
        setSummaryError("");
      } else {
        // Server returned 200 but reported failure (timeout, OpenRouter down,
        // missing key, etc.). Show the friendly fallback in the error slot
        // so the user can retry, without losing the prediction.
        setSummary("");
        setSummaryError(
          response.summary ||
            "AI medical summary is temporarily unavailable. The prediction above is still accurate."
        );
      }
    } catch (error) {
      const errorObj = getApiError(error);
      setSummaryError(errorObj.message);
    } finally {
      setSummaryLoading(false);
    }
  };

  useEffect(() => {
    if (!data || summaryLoading || summary || summaryError) return;
    void loadSummary();
  }, [data, summaryLoading, summary, summaryError]);

  const handleChat = async (question: string) => {
    if (!data || !question.trim()) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question.trim(),
    };

    setChatMessages((current) => [...current, userMessage]);
    setChatInput("");
    setChatLoading(true);
    setChatError("");

    try {
      const response = await chatWithAI({
        prediction: data.prediction!.disease,
        confidence: data.prediction!.confidence,
        disease_information: data.disease_information ?? {},
        herbs: data.recommended_herbs ?? [],
        question: question.trim(),
      });

      setChatMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.answer || "I'm not able to answer that right now.",
        },
      ]);

      toast.success("Chat response received");
    } catch (error) {
      const errorObj = getApiError(error);
      setChatError(errorObj.message);
      toast.error("Failed to get response");
    } finally {
      setChatLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!data || !file) {
      setReportError("A valid image is required to generate the report.");
      toast.error("Missing image for report");
      return;
    }

    setReportLoading(true);
    setReportError("");

    try {
      const blob = await downloadReport(file, {
        prediction: data.prediction,
        disease_info: data.disease_information ?? {},
        herbal_recommendations: data.recommended_herbs ?? [],
        summary: summary || data.message || "No summary available.",
        gradcam_image: data.gradcam_image ?? "",
      });

      const url = window.URL.createObjectURL(blob as Blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "HerbalAI_Report.pdf";
      link.click();
      window.URL.revokeObjectURL(url);

      toast.success("Report downloaded successfully");
    } catch (error) {
      const errorObj = getApiError(error);
      setReportError(errorObj.message);
      toast.error("Failed to download report");
    } finally {
      setReportLoading(false);
    }
  };

  const handleAnalyzeAnother = async (file: File) => {
    try {
      const result = await predictImage(file);

      if (!result.success) {
        toast.error(result.message || "Image validation failed");
        return;
      }

      const dataUrl = await fileToDataUrl(file);

      setData(result);
      setImage(dataUrl);
      setSummary("");
      setSummaryError("");
      setChatMessages([]);
      setChatInput("");
      setChatError("");
      setReportError("");

      sessionStorage.setItem("diagnosis", JSON.stringify(result));
      sessionStorage.setItem("diagnosis-image", dataUrl);
      sessionStorage.setItem("diagnosis-image-data", dataUrl);
      sessionStorage.setItem("diagnosis-file-name", file.name);
      sessionStorage.setItem("diagnosis-file-type", file.type || "image/jpeg");

      setFile(dataUrlToFile(dataUrl, file.name, file.type || "image/jpeg"));
      toast.success("New image analyzed");
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
            Upload a skin image to unlock the premium prediction, summary, and herbal-care experience.
          </p>
          <Link
            href="/diagnose"
            className="mt-8 inline-flex items-center gap-2 rounded-full bg-primary px-5 py-3 text-sm font-medium text-primary-foreground"
          >
            Start diagnosis
          </Link>
        </div>
      </main>
    );
  }

  const isHealthySkin = data.prediction?.disease?.toLowerCase().includes("healthy");
  const herbs = data.recommended_herbs ?? [];
  const predictions = data.top_predictions ?? [];
  const confidence = Number(data.prediction?.confidence) || 0;
  const confidenceLevel = data.prediction?.confidence_level?.toLowerCase() || "medium";
  const condition = data.prediction?.disease || "Unknown";

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
          <p className="text-sm font-medium uppercase tracking-[0.3em] text-primary">AI analysis report</p>
          <h1 className="mt-2 text-4xl font-semibold tracking-tight sm:text-5xl">
            {isHealthySkin ? "Your skin is healthy" : "Your skin insight"}
          </h1>
          <p className="mt-3 max-w-2xl text-lg leading-8 text-muted-foreground">
            {isHealthySkin
              ? "Great news! No visible skin disease was detected. Follow the healthy skincare tips below."
              : "A premium view of the prediction, AI attention map, herbal guidance, and medical summary generated from your image."}
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <Button
            variant="outline"
            size="lg"
            onClick={() => setChatOpen(true)}
            disabled={!data}
            className="h-11 px-5 text-base shadow-sm"
          >
            <MessageCircleMore className="size-5" />
            Open AI chat
          </Button>
          <Button
            size="lg"
            onClick={handleDownloadReport}
            disabled={reportLoading || !file}
            className="h-11 px-5 text-base shadow-sm"
          >
            {reportLoading ? <LoaderCircle className="animate-spin size-5" /> : <Download className="size-5" />}
            {reportLoading ? "Generating report" : "Download AI report"}
          </Button>
        </div>
      </div>

      {reportError && (
        <div className="mt-5">
          <Alert variant="destructive">
            <AlertDescription>{reportError}</AlertDescription>
          </Alert>
        </div>
      )}

      <div className="mt-8 grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
        <div className="space-y-6">
          {/* Main Prediction Card */}
          {isHealthySkin ? (
            <motion.article
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-[2rem] border border-green-200 bg-gradient-to-br from-green-50 to-emerald-50 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.08)]"
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="inline-flex items-center gap-2 rounded-full bg-green-100 px-3 py-1 text-sm font-medium text-green-700">
                    <Check size={16} />
                    Healthy Skin
                  </div>
                  <h2 className="mt-4 text-3xl font-semibold text-green-900">✅ Healthy Skin</h2>
                  <p className="mt-3 text-sm leading-7 text-green-700">
                    No visible skin disease was detected. Your skin appears to be in good condition.
                  </p>
                </div>
                <div className="rounded-2xl border border-green-200 bg-green-100 px-4 py-3 text-right">
                  <p className="text-sm text-green-700">Confidence</p>
                  <p className="mt-1 text-3xl font-semibold text-green-600">{confidence.toFixed(1)}%</p>
                </div>
              </div>

              <div className="mt-6 h-2 overflow-hidden rounded-full bg-green-200">
                <div className="h-full rounded-full bg-green-500" style={{ width: `${Math.min(confidence, 100)}%` }} />
              </div>
            </motion.article>
          ) : (
            <motion.article
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.08)]"
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-medium text-primary">Primary prediction</p>
                  <h2 className="mt-2 text-3xl font-semibold">{condition}</h2>
                  <p className="mt-3 text-sm leading-7 text-muted-foreground">
                    {confidenceLevel.toUpperCase()} confidence level based on the uploaded image.
                  </p>
                </div>
                <div
                  className={`rounded-2xl border px-4 py-3 text-right ${
                    confidenceLevel === "high"
                      ? "border-green-200 bg-green-100"
                      : confidenceLevel === "medium"
                      ? "border-amber-200 bg-amber-100"
                      : "border-red-200 bg-red-100"
                  }`}
                >
                  <p
                    className={`text-sm ${
                      confidenceLevel === "high"
                        ? "text-green-700"
                        : confidenceLevel === "medium"
                        ? "text-amber-700"
                        : "text-red-700"
                    }`}
                  >
                    {confidenceLevel.charAt(0).toUpperCase() + confidenceLevel.slice(1)} Confidence
                  </p>
                  <p
                    className={`mt-1 text-3xl font-semibold ${
                      confidenceLevel === "high"
                        ? "text-green-600"
                        : confidenceLevel === "medium"
                        ? "text-amber-600"
                        : "text-red-600"
                    }`}
                  >
                    {confidence.toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="mt-6 h-2 overflow-hidden rounded-full bg-muted">
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

              <div className="mt-5 flex flex-wrap gap-3 text-sm text-muted-foreground">
                <span className="rounded-full bg-muted px-3 py-1">{confidenceLevel}</span>
                <span className="rounded-full bg-muted px-3 py-1">{data.message ?? "Model output ready"}</span>
              </div>
            </motion.article>
          )}

          {/* Healthy Skin Tips */}
          {isHealthySkin && (
            <motion.article
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="rounded-[2rem] border border-green-200 bg-gradient-to-br from-green-50 to-emerald-50 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.08)]"
            >
              <h3 className="text-xl font-semibold text-green-900">Healthy Skincare Tips</h3>
              <div className="mt-4 space-y-3">
                {healthyTips.map((tip) => (
                  <div key={tip} className="flex items-start gap-3">
                    <Check size={18} className="mt-1 text-green-600 flex-shrink-0" />
                    <p className="text-sm leading-7 text-green-800">{tip}</p>
                  </div>
                ))}
              </div>
            </motion.article>
          )}

          {/* Image Review */}
          <motion.article
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: isHealthySkin ? 0.1 : 0.05 }}
            className="rounded-[2rem] border border-border/70 bg-card/85 p-5 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold">Image review</h3>
              <span className="text-sm text-muted-foreground">
                {gradcamUrl ? "Original image and AI attention map" : "Uploaded image"}
              </span>
            </div>
            <div className={`mt-5 grid gap-4 ${gradcamUrl ? "md:grid-cols-2" : ""}`}>
              <div className="overflow-hidden rounded-[1.5rem] border border-border/70 bg-muted/40">
                <img
                  src={image}
                  alt="Original uploaded image"
                  className="h-72 w-full cursor-zoom-in object-cover"
                  onClick={() => setLightboxImage(image)}
                />
                <div className="p-3 text-sm text-muted-foreground">Original image</div>
              </div>
              {gradcamUrl && (
                <div className="overflow-hidden rounded-[1.5rem] border border-border/70 bg-muted/40">
                  <img
                    src={gradcamUrl}
                    alt="AI attention map"
                    className="h-72 w-full cursor-zoom-in object-cover"
                    onClick={() => setLightboxImage(gradcamUrl)}
                  />
                  <div className="p-3 text-sm text-muted-foreground">AI attention map</div>
                </div>
              )}
            </div>
          </motion.article>

          {/* AI Medical Summary */}
          {!isHealthySkin && (
            <motion.article
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
              className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
            >
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-primary/10 p-2 text-primary">
                  <FileText size={20} />
                </div>
                <div>
                  <h3 className="text-xl font-semibold">AI medical summary</h3>
                  <p className="text-sm text-muted-foreground">Generated automatically after prediction.</p>
                </div>
              </div>

              {summaryLoading ? (
                <div className="mt-5 space-y-3" role="status" aria-live="polite">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <LoaderCircle className="size-4 animate-spin" />
                    Generating AI summary…
                  </div>
                  <div className="h-3 w-full animate-pulse rounded-full bg-muted" />
                  <div className="h-3 w-3/4 animate-pulse rounded-full bg-muted" />
                  <div className="h-3 w-1/2 animate-pulse rounded-full bg-muted" />
                </div>
              ) : summaryError ? (
                <div className="mt-5">
                  <Alert variant="destructive">
                    <AlertDescription className="flex flex-col gap-3">
                      <p>{summaryError}</p>
                      <Button variant="outline" size="sm" onClick={() => void loadSummary()}>
                        Retry summary
                      </Button>
                    </AlertDescription>
                  </Alert>
                </div>
              ) : (
                <p className="mt-5 text-[1rem] leading-8 text-foreground/90">{summary}</p>
              )}
            </motion.article>
          )}

          {/* Analyze Another */}
          <motion.article
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: isHealthySkin ? 0.15 : 0.12 }}
            className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
          >
            <h3 className="text-xl font-semibold mb-4">Analyze Another Image</h3>
            <UploadCard onAnalyze={handleAnalyzeAnother} compact />
          </motion.article>
        </div>

        {/* Right Sidebar */}
        {!isHealthySkin && (
          <div className="space-y-6">
            {/* Disease Information */}
            {data.disease_information && (
              <motion.article
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-2xl bg-primary/10 p-2 text-primary">
                    <Leaf size={20} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold">Disease information</h3>
                    <p className="text-sm text-muted-foreground">What the model identified and how to interpret it.</p>
                  </div>
                </div>
                <div className="mt-6 space-y-4 text-sm leading-8 text-muted-foreground">
                  <InfoBlock title="Description" value={data.disease_information?.description} />
                  <InfoBlock title="Symptoms" value={data.disease_information?.symptoms?.join(" • ")} />
                  <InfoBlock title="Causes" value={data.disease_information?.causes?.join(" • ")} />
                  <InfoBlock title="Prevention" value={data.disease_information?.prevention?.join(" • ")} />
                  <InfoBlock title="When to consult a clinician" value={data.disease_information?.when_to_consult_doctor} />
                </div>
              </motion.article>
            )}

            {/* Top Predictions */}
            {predictions.length > 0 && (
              <motion.article
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: data.disease_information ? 0.12 : 0.1 }}
                className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-2xl bg-primary/10 p-2 text-primary">
                    <Sparkles size={20} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold">Top predictions</h3>
                    <p className="text-sm text-muted-foreground">The model's most likely alternatives.</p>
                  </div>
                </div>
                <div className="mt-5 space-y-4">
                  {predictions.map((item) => (
                    <div key={`${item.disease}-${item.confidence}`}>
                      <div className="mb-2 flex items-center justify-between text-sm">
                        <span className="font-medium">{item.disease}</span>
                        <span className="text-muted-foreground">{item.confidence.toFixed(1)}%</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-muted">
                        <div className="h-full rounded-full bg-primary" style={{ width: `${Math.min(item.confidence, 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </motion.article>
            )}

            {/* Herbal Recommendations */}
            <motion.article
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: data.disease_information || predictions.length > 0 ? 0.14 : 0.1 }}
              className="rounded-[2rem] border border-border/70 bg-card/85 p-6 shadow-[0_20px_70px_rgba(0,0,0,0.06)]"
            >
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-primary/10 p-2 text-primary">
                  <Leaf size={20} />
                </div>
                <div>
                  <h3 className="text-xl font-semibold">Herbal recommendations</h3>
                  <p className="text-sm text-muted-foreground">Educational guidance based on the prediction.</p>
                </div>
              </div>
              <div className="mt-5 space-y-4">
                {herbs.length > 0 ? (
                  herbs.map((herb) => <HerbCard key={herb.name} herb={herb} details={data.herb_details?.[herb.name]} />)
                ) : (
                  <div className="rounded-[1.2rem] border border-border/70 bg-muted/40 p-4 text-sm text-muted-foreground">
                    No herbal recommendations available.
                  </div>
                )}
              </div>
            </motion.article>
          </div>
        )}
      </div>

      {/* Chat Sheet */}
      <Sheet open={chatOpen} onOpenChange={setChatOpen}>
        <SheetContent side="right" className="flex h-full w-full max-w-xl flex-col border-border/70 bg-background/95 p-0">
          <SheetHeader className="border-b border-border/60 p-6">
            <SheetTitle className="flex items-center gap-2 text-xl">
              <Bot size={18} className="text-primary" />
              Herbal-AI assistant
            </SheetTitle>
            <SheetDescription>
              {data ? "Ask follow-up questions about the diagnosis and herbal recommendations." : "Please analyze an image first."}
            </SheetDescription>
          </SheetHeader>

          <div className="flex-1 space-y-4 overflow-y-auto p-6">
            {chatMessages.length === 0 && !chatLoading && (
              <div className="rounded-[1.2rem] border border-border/70 bg-muted/40 p-4 text-sm text-muted-foreground">
                The assistant already knows the diagnosis context and can answer questions about the condition, herbs, and next steps.
              </div>
            )}

            {chatMessages.map((message) => (
              <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-[1.15rem] px-4 py-3 text-sm leading-7 ${
                    message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted/70 text-foreground"
                  }`}
                >
                  {message.content}
                </div>
              </div>
            ))}

            {chatLoading && (
              <div className="flex justify-start">
                <div className="rounded-[1.15rem] bg-muted/70 px-4 py-3 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <LoaderCircle className="animate-spin" size={16} />
                    Thinking…
                  </div>
                </div>
              </div>
            )}

            {chatError && (
              <div className="rounded-[1.2rem] border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
                {chatError}
              </div>
            )}
          </div>

          <div className="border-t border-border/60 p-6">
            <div className="flex flex-wrap gap-2">
              {suggestions.map((item) => (
                <button
                  key={item}
                  onClick={() => void handleChat(item)}
                  disabled={!data || chatLoading}
                  className="rounded-full border border-border/70 bg-background px-3 py-2 text-sm text-muted-foreground transition hover:border-primary hover:text-foreground disabled:opacity-50"
                >
                  {item}
                </button>
              ))}
            </div>
            <div className="mt-4 flex gap-2">
              <input
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !chatLoading && data) {
                    event.preventDefault();
                    void handleChat(chatInput);
                  }
                }}
                placeholder={data ? "Ask the AI assistant" : "Please analyze an image first"}
                disabled={!data || chatLoading}
                className="flex-1 rounded-full border border-border/70 bg-background px-4 py-3 text-sm outline-none focus:border-primary disabled:opacity-50"
              />
              <Button onClick={() => void handleChat(chatInput)} disabled={chatLoading || !chatInput.trim() || !data}>
                Send
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </main>
  );
}

function InfoBlock({ title, value }: { title: string; value?: string }) {
  return (
    <div>
      <p className="font-semibold text-foreground">{title}</p>
      <p className="mt-1">{value || "Not provided by the current recommendation."}</p>
    </div>
  );
}

function HerbCard({
  herb,
  details,
}: {
  herb: Herb;
  details?: { benefits?: string[]; preparation_method?: string; side_effects?: string[]; contraindications?: string[] };
}) {
  const list = (items?: string[]) => items?.filter(Boolean).join(" • ") || "Not provided by the current recommendation.";

  return (
    <article className="rounded-[1.3rem] border border-border/70 bg-muted/30 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.25em] text-primary">{herb.evidence_level || "Reference"} evidence</p>
          <h4 className="mt-2 text-lg font-semibold">{herb.name}</h4>
          <p className="mt-1 text-sm text-muted-foreground">{herb.botanical_name || "Botanical name unavailable"}</p>
        </div>
        {typeof herb.efficacy === "number" && (
          <div className="rounded-full bg-primary/10 px-3 py-1 text-sm font-medium text-primary">{herb.efficacy}%</div>
        )}
      </div>
      <div className="mt-4 space-y-3 text-sm leading-7 text-muted-foreground">
        <div>
          <p className="font-medium text-foreground">Benefits</p>
          <p>{list(details?.benefits ?? herb.benefits)}</p>
        </div>
        <div>
          <p className="font-medium text-foreground">Preparation</p>
          <p>{details?.preparation_method ?? herb.preparation_method ?? "Not provided by the current recommendation."}</p>
        </div>
        <div>
          <p className="font-medium text-foreground">Precautions</p>
          <p>{list([...(details?.side_effects ?? herb.side_effects ?? []), ...(details?.contraindications ?? [])])}</p>
        </div>
      </div>
    </article>
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
