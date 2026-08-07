import axios from "axios";
import type { ChatResponse, PredictionResponse, SummaryResponse, HerbPredictionResponse } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60_000,
});

// ── Auth Interceptor ──────────────────────────────────────
// Automatically attaches the Clerk JWT token to every request.

api.interceptors.request.use(async (config) => {
  const token = await getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Token helpers ──────────────────────────────────────────
// Tokens are managed by Clerk, not localStorage.

export async function getAccessToken(): Promise<string | null> {
  if (typeof window === "undefined") return null;
  try {
    const { getToken } = await import("@clerk/nextjs");
    const token = await getToken();
    return token ?? null;
  } catch {
    return null;
  }
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
  const { signOut } = await import("@clerk/nextjs");
  await signOut();
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

export function getApiError(error: unknown): { message: string; status?: number } {
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
      return {
        message: "Authentication required. Please refresh the page and try again.",
        status,
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
