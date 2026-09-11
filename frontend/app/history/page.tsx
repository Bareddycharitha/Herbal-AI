"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertCircle,
  Calendar,
  Clock,
  ExternalLink,
  History,
  Info,
  Leaf,
  Loader2,
  LogIn,
  Search,
  Sparkles,
  Trash2,
  ChevronRight,
  ShieldCheck,
  FileText,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useAuth } from "@/providers/AuthProvider";
import {
  clearPredictionHistory,
  deleteHistoryItem,
  getApiError,
  getPredictionHistory,
} from "@/lib/api";
import type { HistoryRecord } from "@/types";
import { toast } from "sonner";

export default function HistoryPage() {
  const router = useRouter();
  const { isAuthenticated, tokenReady, user } = useAuth();

  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRecord, setSelectedRecord] = useState<HistoryRecord | null>(null);
  const [recordToDelete, setRecordToDelete] = useState<string | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Fetch history once user is authenticated and token is ready
  useEffect(() => {
    if (!tokenReady) return;

    if (!isAuthenticated) {
      setLoading(false);
      return;
    }

    const fetchHistory = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await getPredictionHistory(100, 0);
        if (response.success) {
          setHistory(response.history || []);
        } else {
          setError("Failed to fetch history.");
        }
      } catch (err) {
        const errorObj = getApiError(err);
        setError(errorObj.message);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [isAuthenticated, tokenReady]);

  const getPredictionName = (pred: unknown): string => {
    if (!pred) return "Skin Assessment";
    if (typeof pred === "string") return pred.replace(/_/g, " ");
    if (typeof pred === "object" && pred !== null) {
      const p = pred as Record<string, unknown>;
      const name = p.disease || p.herb || p.class || p.name || "Skin Assessment";
      return String(name).replace(/_/g, " ");
    }
    return String(pred);
  };

  const filteredHistory = useMemo(() => {
    if (!searchQuery.trim()) return history;
    const query = searchQuery.toLowerCase();
    return history.filter((item) => {
      const pred = getPredictionName(item.prediction).toLowerCase();
      const summary = (item.ai_summary || "").toLowerCase();
      const herbs = (item.recommended_herbs || [])
        .map((h) => h.name.toLowerCase())
        .join(" ");
      return pred.includes(query) || summary.includes(query) || herbs.includes(query);
    });
  }, [history, searchQuery]);

  const handleDeleteItem = async () => {
    if (!recordToDelete) return;
    setIsDeleting(true);
    try {
      await deleteHistoryItem(recordToDelete);
      setHistory((prev) => prev.filter((item) => item.id !== recordToDelete));
      if (selectedRecord?.id === recordToDelete) {
        setSelectedRecord(null);
      }
      toast.success("Assessment record deleted successfully");
    } catch (err) {
      const errorObj = getApiError(err);
      toast.error(errorObj.message || "Failed to delete record");
    } finally {
      setIsDeleting(false);
      setRecordToDelete(null);
    }
  };

  const handleClearAll = async () => {
    setIsDeleting(true);
    try {
      await clearPredictionHistory();
      setHistory([]);
      setSelectedRecord(null);
      toast.success("All assessment history cleared");
    } catch (err) {
      const errorObj = getApiError(err);
      toast.error(errorObj.message || "Failed to clear history");
    } finally {
      setIsDeleting(false);
      setShowClearConfirm(false);
    }
  };

  const loadIntoResults = (item: HistoryRecord) => {
    const numConf = typeof item.confidence === "number" ? item.confidence : parseFloat(String(item.confidence)) || 0;
    // Construct PredictionResponse-compatible structure to view in results
    const fullPrediction = {
      success: true,
      message: "Loaded from analysis history",
      prediction_id: item.prediction_id || item.id,
      prediction: {
        disease: getPredictionName(item.prediction),
        confidence: numConf,
        confidence_level: item.confidence_level || (numConf >= 70 ? "high" : numConf >= 40 ? "medium" : "low"),
      },
      top_predictions: item.top_predictions?.map((tp) => ({
        disease: tp.disease || tp.herb || tp.class || "Unknown",
        confidence: tp.confidence,
      })),
      disease_information: item.disease_information,
      recommended_herbs: item.recommended_herbs,
      ai_summary: item.ai_summary,
    };

    sessionStorage.setItem("diagnosis", JSON.stringify(fullPrediction));
    if (item.image_path) {
      sessionStorage.setItem("diagnosis-image", item.image_path);
    }
    router.push("/results");
  };

  const formatDate = (isoString?: string) => {
    if (!isoString) return "Recent";
    try {
      const date = new Date(isoString);
      return new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      }).format(date);
    } catch {
      return isoString;
    }
  };

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 75) {
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          {confidence.toFixed(1)}% High
        </span>
      );
    }
    if (confidence >= 45) {
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/10 px-3 py-1 text-xs font-semibold text-amber-600 dark:text-amber-400 border border-amber-500/20">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
          {confidence.toFixed(1)}% Moderate
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-500/10 px-3 py-1 text-xs font-semibold text-rose-600 dark:text-rose-400 border border-rose-500/20">
        <span className="h-1.5 w-1.5 rounded-full bg-rose-500" />
        {confidence.toFixed(1)}% Low
      </span>
    );
  };

  if (!tokenReady) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <main className="mx-auto max-w-4xl px-5 py-20 text-center sm:px-8">
        <div className="mx-auto max-w-md rounded-[2.5rem] border border-border/80 bg-card/70 p-10 shadow-2xl backdrop-blur-xl">
          <div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-primary/10 text-primary">
            <History className="h-8 w-8" />
          </div>
          <h1 className="mt-6 text-2xl font-bold tracking-tight sm:text-3xl">
            Assessment History
          </h1>
          <p className="mt-3 text-muted-foreground">
            Sign in to view your past AI skin analyses, diagnostic summaries, and herbal recommendations saved securely to your account.
          </p>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:justify-center">
            <Link href="/login">
              <Button className="w-full sm:w-auto h-12 px-6">
                <LogIn className="mr-2 h-4 w-4" />
                Sign In to Account
              </Button>
            </Link>
            <Link href="/diagnose">
              <Button variant="outline" className="w-full sm:w-auto h-12 px-6">
                Start Analysis
              </Button>
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-7xl px-5 py-12 sm:px-8 lg:px-10">
      {/* Page Header */}
      <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3.5 py-1 text-xs font-semibold uppercase tracking-wider text-primary">
            <History size={14} />
            Patient Records
          </div>
          <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
            Analysis History
          </h1>
          <p className="mt-2 text-muted-foreground">
            Review your past skin health scans, herbal remedies, and medical explanations.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {history.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowClearConfirm(true)}
              className="text-destructive hover:bg-destructive/10 hover:text-destructive h-10 px-4"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Clear All
            </Button>
          )}
          <Link href="/diagnose">
            <Button className="h-10 px-5 shadow-md">
              <Sparkles className="mr-2 h-4 w-4" />
              New Analysis
            </Button>
          </Link>
        </div>
      </div>

      {/* Filter and Search Bar */}
      {history.length > 0 && (
        <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by condition, herb, or summary keywords..."
              className="pl-10 h-11 rounded-2xl bg-card/60 border-border/70"
            />
          </div>
          <div className="text-sm text-muted-foreground whitespace-nowrap">
            Showing <strong className="text-foreground">{filteredHistory.length}</strong> of {history.length} assessments
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <Alert variant="destructive" className="mt-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Content Area */}
      {loading ? (
        <div className="mt-16 flex flex-col items-center justify-center gap-3 text-muted-foreground">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm">Loading your assessment history...</p>
        </div>
      ) : history.length === 0 ? (
        <div className="mt-12 rounded-[2.5rem] border border-dashed border-border/80 bg-card/30 p-12 text-center backdrop-blur-sm">
          <div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-muted text-muted-foreground">
            <Activity className="h-8 w-8" />
          </div>
          <h2 className="mt-5 text-xl font-semibold">No assessments found yet</h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
            You haven't run any skin analyses while logged in. When you perform an analysis, your clinical reports and herbal plans will automatically appear here.
          </p>
          <div className="mt-6">
            <Link href="/diagnose">
              <Button className="h-11 px-6">
                Start Your First Assessment
              </Button>
            </Link>
          </div>
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="mt-12 rounded-2xl border border-border/70 bg-card/30 p-8 text-center text-muted-foreground">
          <p>No assessments match your search query &quot;{searchQuery}&quot;.</p>
          <Button
            variant="link"
            onClick={() => setSearchQuery("")}
            className="mt-2 text-primary"
          >
            Clear Search
          </Button>
        </div>
      ) : (
        <div className="mt-8 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          <AnimatePresence>
            {filteredHistory.map((item, idx) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.2, delay: idx * 0.03 }}
                className="group relative flex flex-col justify-between rounded-[2rem] border border-border/70 bg-card/75 p-6 shadow-sm transition-all hover:border-primary/40 hover:shadow-xl hover:bg-card/90"
              >
                <div>
                  {/* Top row: Date & Delete */}
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="flex items-center gap-1.5">
                      <Calendar size={13} />
                      {formatDate(item.created_at)}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setRecordToDelete(item.id);
                      }}
                      className="rounded-lg p-1 text-muted-foreground transition hover:bg-destructive/10 hover:text-destructive"
                      title="Delete record"
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>

                  {/* Condition & Confidence */}
                  <div className="mt-4">
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="text-xl font-bold capitalize text-foreground tracking-tight">
                        {getPredictionName(item.prediction)}
                      </h3>
                      {getConfidenceBadge(item.confidence)}
                    </div>
                  </div>

                  {/* AI Summary Snippet */}
                  {item.ai_summary ? (
                    <div className="mt-3.5 rounded-xl bg-primary/5 p-3 text-xs leading-relaxed text-muted-foreground border border-primary/10">
                      <div className="flex items-center gap-1.5 font-medium text-primary mb-1">
                        <Sparkles size={12} />
                        <span>AI Medical Summary</span>
                      </div>
                      <p className="line-clamp-3">{item.ai_summary}</p>
                    </div>
                  ) : item.disease_information?.description ? (
                    <p className="mt-3 text-xs text-muted-foreground line-clamp-2">
                      {item.disease_information.description}
                    </p>
                  ) : null}

                  {/* Recommended Herbs */}
                  {item.recommended_herbs && item.recommended_herbs.length > 0 && (
                    <div className="mt-4">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground mb-2">
                        <Leaf size={13} className="text-primary" />
                        <span>Recommended Herbs</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {item.recommended_herbs.slice(0, 3).map((herb, hIdx) => (
                          <span
                            key={hIdx}
                            className="inline-flex items-center rounded-lg bg-muted px-2.5 py-1 text-[11px] font-medium text-foreground"
                          >
                            {herb.name}
                          </span>
                        ))}
                        {item.recommended_herbs.length > 3 && (
                          <span className="inline-flex items-center rounded-lg bg-muted/60 px-2 py-1 text-[11px] text-muted-foreground">
                            +{item.recommended_herbs.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Card Actions */}
                <div className="mt-6 flex items-center gap-2 pt-4 border-t border-border/50">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 rounded-xl h-9 text-xs"
                    onClick={() => setSelectedRecord(item)}
                  >
                    <Info className="mr-1.5 h-3.5 w-3.5" />
                    Quick Details
                  </Button>
                  <Button
                    size="sm"
                    className="flex-1 rounded-xl h-9 text-xs"
                    onClick={() => loadIntoResults(item)}
                  >
                    <span>Open Report</span>
                    <ExternalLink className="ml-1.5 h-3.5 w-3.5" />
                  </Button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Record Quick Details Dialog */}
      <Dialog
        open={!!selectedRecord}
        onOpenChange={(open) => !open && setSelectedRecord(null)}
      >
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto rounded-[2rem] p-7">
          {selectedRecord && (
            <div>
              <DialogHeader>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                    <Calendar size={13} />
                    {formatDate(selectedRecord.created_at)}
                  </span>
                  {getConfidenceBadge(selectedRecord.confidence)}
                </div>
                <DialogTitle className="text-2xl font-bold capitalize mt-2">
                  {getPredictionName(selectedRecord.prediction)}
                </DialogTitle>
                <DialogDescription className="text-sm">
                  Full assessment overview and herbal formulation breakdown.
                </DialogDescription>
              </DialogHeader>

              {/* AI Summary Section */}
              {selectedRecord.ai_summary && (
                <div className="mt-5 rounded-2xl bg-primary/5 p-4 border border-primary/10">
                  <div className="flex items-center gap-2 text-sm font-semibold text-primary mb-2">
                    <Sparkles size={16} />
                    <span>AI Clinical Summary</span>
                  </div>
                  <p className="text-sm leading-relaxed text-foreground whitespace-pre-line">
                    {selectedRecord.ai_summary}
                  </p>
                </div>
              )}

              {/* Disease Info */}
              {selectedRecord.disease_information && (
                <div className="mt-5 space-y-3">
                  {selectedRecord.disease_information.description && (
                    <div>
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        About Condition
                      </h4>
                      <p className="mt-1 text-sm text-foreground">
                        {selectedRecord.disease_information.description}
                      </p>
                    </div>
                  )}

                  {selectedRecord.disease_information.symptoms &&
                    selectedRecord.disease_information.symptoms.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          Common Symptoms
                        </h4>
                        <ul className="mt-1.5 grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-sm">
                          {selectedRecord.disease_information.symptoms.map((sym, sIdx) => (
                            <li key={sIdx} className="flex items-start gap-2">
                              <span className="text-primary mt-0.5">•</span>
                              <span>{sym}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                </div>
              )}

              {/* Recommended Herbs */}
              {selectedRecord.recommended_herbs &&
                selectedRecord.recommended_herbs.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5 mb-3">
                      <Leaf size={14} className="text-primary" />
                      Recommended Herbal Formulations
                    </h4>
                    <div className="grid gap-2.5">
                      {selectedRecord.recommended_herbs.map((herb, hIdx) => (
                        <div
                          key={hIdx}
                          className="rounded-xl border border-border/70 bg-muted/30 p-3.5 text-sm"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-foreground">
                              {herb.name}
                            </span>
                            {herb.botanical_name && (
                              <span className="text-xs italic text-muted-foreground">
                                {herb.botanical_name}
                              </span>
                            )}
                          </div>
                          {herb.benefits && herb.benefits.length > 0 && (
                            <p className="mt-1.5 text-xs text-muted-foreground">
                              <strong>Benefits:</strong> {herb.benefits.join(", ")}
                            </p>
                          )}
                          {herb.preparation_method && (
                            <p className="mt-1 text-xs text-muted-foreground">
                              <strong>Application:</strong> {herb.preparation_method}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              <DialogFooter className="mt-8 flex gap-3">
                <Button
                  variant="outline"
                  onClick={() => setSelectedRecord(null)}
                  className="rounded-xl"
                >
                  Close
                </Button>
                <Button
                  onClick={() => loadIntoResults(selectedRecord)}
                  className="rounded-xl"
                >
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Open Full Results Page
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={!!recordToDelete}
        onOpenChange={(open) => !open && setRecordToDelete(null)}
      >
        <DialogContent className="max-w-md rounded-2xl p-6">
          <DialogHeader>
            <DialogTitle>Delete Assessment Record?</DialogTitle>
            <DialogDescription>
              Are you sure you want to remove this prediction from your history? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-6 flex gap-2">
            <Button
              variant="outline"
              onClick={() => setRecordToDelete(null)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteItem}
              disabled={isDeleting}
            >
              {isDeleting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete Record"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Clear All Confirmation Dialog */}
      <Dialog
        open={showClearConfirm}
        onOpenChange={(open) => !open && setShowClearConfirm(false)}
      >
        <DialogContent className="max-w-md rounded-2xl p-6">
          <DialogHeader>
            <DialogTitle>Clear All History?</DialogTitle>
            <DialogDescription>
              This will permanently delete all your saved skin analysis records from the database.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-6 flex gap-2">
            <Button
              variant="outline"
              onClick={() => setShowClearConfirm(false)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleClearAll}
              disabled={isDeleting}
            >
              {isDeleting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Clearing...
                </>
              ) : (
                "Clear Everything"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  );
}
