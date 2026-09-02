"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react";
import {
  useAuth as useClerkAuth,
  useUser as useClerkUser,
  ClerkProvider,
} from "@clerk/nextjs";
import type { AuthTokens, User } from "@/types";

interface AuthContextType {
  user: User | null;
  tokens: AuthTokens | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  /**
   * True once the Clerk session has hydrated AND the axios token getter
   * has been installed in lib/api.ts. Consumers that need to fire
   * authenticated requests on mount should wait for this — calling
   * them earlier races the Clerk hydration effect and produces 401s.
   */
  tokenReady: boolean;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

function AuthProviderInner({ children }: { children: ReactNode }) {
  const { isSignedIn, getToken, signOut } = useClerkAuth();
  const { user: clerkUser } = useClerkUser();
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  // Tracks whether the axios token getter has been installed. The
  // getter is installed synchronously the moment we know the user is
  // signed in — it does NOT wait on a network round-trip to Clerk.
  // The actual token is resolved lazily inside the getter (which
  // calls Clerk's getToken() per request, with skipCache: true so it
  // is always fresh). Components that need to fire an authenticated
  // request can wait on tokenReady without blocking the rest of the
  // app on Clerk's network latency.
  const [tokenReady, setTokenReady] = useState(false);

  useEffect(() => {
    if (!isSignedIn || !clerkUser) {
      setUser(null);
      setTokens(null);
      setTokenReady(false);
      setIsLoading(false);
      return;
    }

    // Populate local user state from Clerk's already-loaded user data
    // (synchronous — no network round-trip). The backend's /auth/me
    // profile call still happens in the background for normalisation,
    // but its outcome is decoupled from isLoading so a slow /auth/me
    // cannot keep the whole page stuck on a loading state.
    setIsLoading(false);
    setUser({
      id: clerkUser.id,
      email: clerkUser.emailAddresses[0]?.emailAddress ?? "",
      full_name:
        clerkUser.fullName ??
        clerkUser.firstName ??
        clerkUser.lastName ??
        "",
      role: "user",
      is_active: true,
    });
    setTokens({
      access_token: "clerk", // placeholder; real token resolved per-request by the getter
      refresh_token: "",
      expires_in: 0,
      token_type: "bearer",
    });

    // Install the token getter synchronously. The getter awaits
    // getToken({ skipCache: true }) on every call so the backend
    // always sees a fresh, valid JWT. We do NOT await the first
    // getToken() call here — that would re-introduce the very race
    // we are removing (isLoading stuck while Clerk's network call
    // is in flight).
    let cancelled = false;
    void (async () => {
      try {
        const { setTokenGetter } = await import("@/lib/api");
        if (cancelled) return;
        setTokenGetter(() => getToken({ skipCache: true }));
        if (!cancelled) setTokenReady(true);
      } catch (error) {
        console.error("Failed to install token getter:", error);
      }
    })();

    // Best-effort backend profile normalisation. Runs in the
    // background; failures fall back to the Clerk-derived user.
    void (async () => {
      try {
        const token = await getToken({ skipCache: true });
        if (!token || cancelled) return;
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/v1/auth/me`,
          { headers: { Authorization: `Bearer ${token}` } },
        );
        if (cancelled) return;
        if (response.ok) {
          const userData = await response.json();
          if (!cancelled) setUser(userData);
        }
      } catch (error) {
        console.error("Failed to normalise user profile:", error);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [isSignedIn, clerkUser, getToken]);

  async function logout() {
    await signOut();
    setUser(null);
    setTokens(null);
    setTokenReady(false);
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        tokens,
        isLoading,
        isAuthenticated: !!user && !!isSignedIn,
        tokenReady,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function AuthProvider({ children }: { children: ReactNode }) {
  return (
    <ClerkProvider>
      <AuthProviderInner>{children}</AuthProviderInner>
    </ClerkProvider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
