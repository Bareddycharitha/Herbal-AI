import axios from "axios";
import type { ChatResponse, PredictionResponse, SummaryResponse, HerbPredictionResponse } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60_000,
});

// Token getter function - will be set by AuthProvider
let getTokenFn: (() => Promise<string | null>) | null = null;

// Setter for token getter function
export const setTokenGetter = (fn: () => Promise<string | null>) => {
  getTokenFn = fn;
};

// ── Auth Interceptor ──────────────────────────
// Automatically attaches the Clerk JWT token to every request.
api.interceptors.request.use(async (config) => {
  // Skip token attachment for requests to auth endpoints (to avoid circular issues)

  const token = getTokenFn ? await getTokenFn() : null;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
    // Diagnostic: log the first request that carries a token so we
    // can confirm the Authorization header is actually being sent.
    if (typeof console !== "undefined" && !(config as any).__authLogged) {
      (config as any).__authLogged = true;
      console.debug("[Herbal-AI] attaching Bearer token to", {
        method: config.method,
        url: config.url,
        token_chars: token.length,
      });
    }
  } else if (typeof console !== "undefined") {
    // No token getter installed or getToken returned null. The request
    // will hit the backend with no Authorization header. Log this
    // prominently so the cause is obvious in the console.
    console.warn("[Herbal-AI] sending UNAUTHENTICATED request to", {
      method: config.method,
      url: config.url,
      getter_installed: !!getTokenFn,
    });
  }
  return config;
});

// ── Token helpers ──────────────────────────────────────────
// Tokens are managed by Clerk, not localStorage.

export async function getAccessToken(): Promise<string | null> {
  if (typeof window === "undefined") return null;
  return getTokenFn ? await getTokenFn() : null;
}

export function getRefreshToken(): string | null {
  // Clerk manages refresh tokens internally; not exposed to the client.
  return null;
}

export function setTokens(_accessToken: string, _refreshToken: string): void {
  // Tokens are managed by Clerk, no manual storage needed.
}

export function clearTokens(): void {
  // Tokens are managed by Clerk, no manual clearing needed.
}

export async function isAuthenticated(): Promise<boolean> {
  try {
    const token = await getAccessToken();
    return !!token;
  } catch {
    return false;
  }
}

// ── Auth API ───────────────────────────────────────────────
// Clerk handles signup, login, logout, email verification,
// password reset, and Google OAuth on the frontend.
// The backend only provides profile management endpoints.

export async function login(_email: string, _password: string) {
  // Clerk handles authentication on the frontend.
  // This function is kept for API compatibility but
  // delegates to Clerk's sign-in flow.
  throw new Error("Use Clerk's SignIn component for login");
}

export async function register(
  _email: string,
  _password: string,
  _fullName?: string
) {
  // Clerk handles registration on the frontend.
  // This function is kept for API compatibility but
  // delegates to Clerk's sign-up flow.
  throw new Error("Use Clerk's SignUp component for registration");
}

export async function logout() {
  // Clerk logout is handled by AuthProvider using useAuth().signOut().
  // This function is retained only for API compatibility.
  throw new Error("Use the logout function from AuthProvider instead.");
}
export function getApiBaseUrl() {
  return API_BASE;
}

export async function predictImage(image: File) {
  const formData = new FormData();
  formData.append("image", image);
  const { data } = await api.post<PredictionResponse>("/api/v1/predict/", formData);
  return data;
}

export type GradcamStatus =
  | { prediction_id: string; ready: true; status: "ready"; url: string; error?: string }
  | { prediction_id: string; ready: false; status: "pending" | "no_gradcam" | "failed"; url: null; error?: string };

export async function fetchGradcamStatus(
  predictionId: string,
  waitSeconds = 0,
): Promise<GradcamStatus> {
  const { data } = await api.get<GradcamStatus>(
    `/api/v1/gradcam/${encodeURIComponent(predictionId)}`,
    { params: { wait_seconds: waitSeconds } },
  );
  return data;
}

export async function generateSummary(payload: {
  prediction: string;
  confidence: number;
  disease_information: Record<string, unknown>;
  herbs: Array<Record<string, unknown>>;
}) {
  const { data } = await api.post<SummaryResponse>("/api/v1/summary/", payload);
  return data;
}

export async function chatWithAI(payload: {
  prediction: string;
  confidence: number;
  disease_information: Record<string, unknown>;
  herbs: Array<Record<string, unknown>>;
  question: string;
}) {
  const { data } = await api.post<ChatResponse>("/api/v1/chat/", payload);
  return data;
}

export function getApiError(error: unknown): { message: string; status?: number; reason?: string } {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status;
    const message =
      error.response?.data?.detail ??
      error.response?.data?.message ??
      error.message ??
      "We couldn't complete that request. Please try again.";

    // Handle specific HTTP errors
    if (status === 400) {
      return {
        message: message || "Invalid request. Please check your input.",
        status,
      };
    }
    if (status === 401) {
      // Diagnostic: surface the failing endpoint + specific reason in
      // the console. The backend no longer enforces the token's exp
      // claim, so a 401 here is a real authentication problem
      // (missing token, signature failure, JWKS mismatch) — not a
      // "session expired" prompt. We don't ask the user to sign in
      // again; we report the problem in the toast and let the user
      // try the action once more.
      const errorBody = error.response?.data;
      const reason =
        errorBody?.error?.details?.reason ??
        errorBody?.detail ??
        "unauthorized";
      const detail = errorBody?.error?.details?.detail;

      if (typeof console !== "undefined") {
        console.warn(
          "[Herbal-AI] 401 from protected endpoint",
          {
            method: error.config?.method,
            url: error.config?.url,
            status,
            reason,
            detail,
            body: errorBody,
          },
        );
      }

      // Map known reasons to user-friendly messages. Anything we don't
      // recognise falls back to the generic "could not authorize" copy
      // — we deliberately do NOT tell the user to re-login, because
      // the token's exp is no longer enforced. If this fires, the
      // problem is with the token itself, not the user's session.
      const reasonToMessage: Record<string, string> = {
        no_matching_jwks_key:
          "We couldn't authorize this request. Please try again.",
        token_invalid:
          "We couldn't authorize this request. Please try again.",
        token_verification_error:
          "We couldn't authorize this request. Please try again.",
      };
      return {
        message:
          reasonToMessage[reason] ??
          "We couldn't authorize this request. Please try again.",
        status,
        reason,
      };
    }
    if (status === 503) {
      // 503 is the structured "model checkpoint missing" / "service
      // unavailable" response. Pull a friendly message from the
      // backend's error body when we recognise the code, otherwise
      // fall back to the generic copy.
      const errorBody = error.response?.data;
      const code = errorBody?.error?.code;
      const codeToMessage: Record<string, string> = {
        model_load_error:
          "The AI model isn't ready yet — its training checkpoint is missing on the server. Please contact the project maintainer or check the README for how to download model weights.",
        service_unavailable:
          "The backend is temporarily unavailable. Please try again in a moment.",
      };
      const detail = errorBody?.error?.details;
      const detailSuffix =
        detail && typeof detail === "object" && typeof detail.model_path === "string"
          ? ` (missing: ${detail.model_path})`
          : "";
      const friendly =
        (code && codeToMessage[code]) ??
        "The backend is temporarily unavailable. Please try again in a moment.";
      return {
        message: friendly + detailSuffix,
        status,
        reason: code,
      };
    }
    if (status === 404) {
      return {
        message: "Resource not found.",
        status,
      };
    }
    if (status === 422) {
      return {
        message: message || "Invalid input. Please check the data.",
        status,
      };
    }
    if (status === 500) {
      return {
        message: "Server error. Please try again later.",
        status,
      };
    }
    if (status === 503) {
      return {
        message: "Server unavailable. Please try again later.",
        status,
      };
    }

    return { message, status };
  }

  if (error instanceof Error) {
    if (error.message.includes("timeout")) {
      return {
        message: "Request timed out. Please check your connection and try again.",
      };
    }
    return { message: error.message };
  }

  return {
    message: "Something went wrong. Please try again.",
  };
}

export async function downloadReport(image: File, reportData: object) {
  const formData = new FormData();

  formData.append("image", image);
  formData.append("report_data", JSON.stringify(reportData));

  const response = await api.post("/api/v1/report/", formData, {
    responseType: "blob",
  });

  return response.data;
}

export async function predictHerb(image: File) {
  const formData = new FormData();
  formData.append("image", image);
  const { data } = await api.post<{ success?: boolean; message?: string; prediction?: { herb: string; confidence: number; top_predictions?: { class: string; confidence: number }[]; is_confident?: boolean }; knowledge?: Record<string, unknown> | null }>("/api/v1/herb/", formData);

  // Handle backend validation errors (e.g., skin image uploaded in herb module)
  if (data.success === false) {
    return {
      success: false,
      message: data.message || "Image validation failed. Please try another image.",
    };
  }

  const knowledge = data.knowledge ?? {};
  return {
    success: true, message: "Herb identified successfully.",
    prediction: { herb: data.prediction!.herb, confidence: data.prediction!.confidence, confidence_level: data.prediction!.is_confident ? "high" : "low" },
    top_predictions: (data.prediction!.top_predictions ?? []).map((item) => ({ herb: item.class, confidence: item.confidence })),
    herb_information: {
      common_name: typeof knowledge.name === "string" ? knowledge.name : data.prediction!.herb,
      scientific_name: typeof knowledge.botanical_name === "string" ? knowledge.botanical_name : undefined,
      family: typeof knowledge.family === "string" ? knowledge.family : undefined,
      medicinal_properties: Array.isArray(knowledge.benefits) ? knowledge.benefits.filter((item): item is string => typeof item === "string") : [],
      preparation_methods: typeof knowledge.preparation_method === "string" ? [knowledge.preparation_method] : [],
      skin_conditions_supported: Array.isArray(knowledge.skin_types) ? knowledge.skin_types.filter((item): item is string => typeof item === "string") : [],
      active_compounds: Array.isArray(knowledge.active_compounds) ? knowledge.active_compounds.filter((item): item is string => typeof item === "string") : [],
      precautions: [...(Array.isArray(knowledge.side_effects) ? knowledge.side_effects : []), ...(Array.isArray(knowledge.contraindications) ? knowledge.contraindications : [])].filter((item): item is string => typeof item === "string"),
    },
  } satisfies HerbPredictionResponse;
}
