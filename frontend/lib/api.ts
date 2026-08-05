import axios from "axios";
import type { ChatResponse, PredictionResponse, SummaryResponse, HerbPredictionResponse } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60_000,
});

// ── Auth Interceptor ──────────────────────────────────────
// Automatically attaches the access token to every request.

api.interceptors.request.use((config) => {
  const token = getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Token helpers ──────────────────────────────────────────

const ACCESS_TOKEN_KEY = "herb_ai_access_token";
const REFRESH_TOKEN_KEY = "herb_ai_refresh_token";

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function setTokens(accessToken: string, refreshToken: string): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
}

export function clearTokens(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

export function isAuthenticated(): boolean {
  return !!getAccessToken();
}

// ── Auth API ───────────────────────────────────────────────

export async function login(email: string, password: string) {
  const { data } = await api.post<{
    access_token: string;
    refresh_token: string;
    expires_in: number;
    token_type: string;
  }>("/api/v1/auth/login/json", { email, password });

  setTokens(data.access_token, data.refresh_token);
  return data;
}

export async function register(
  email: string,
  password: string,
  fullName?: string
) {
  const { data } = await api.post<{
    id: number;
    email: string;
    full_name: string;
    role: string;
    is_active: boolean;
  }>("/api/v1/auth/register", { email, password, full_name: fullName });

  return data;
}

export async function logout() {
  clearTokens();
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
