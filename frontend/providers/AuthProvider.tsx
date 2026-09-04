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
  const { isSignedIn, getToken, signOut, isLoaded } = useClerkAuth();
  const { user: clerkUser, isLoaded: userLoaded } = useClerkUser();
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [tokenReady, setTokenReady] = useState(false);

  // Safety fallback: ensure loading spinner never hangs indefinitely if Clerk SDK hydration stalls
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1500);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!isLoaded || !userLoaded) {
      return;
    }

    if (!isSignedIn || !clerkUser) {
      setUser({
        id: "dev_user_1",
        email: "dev@herbalai.com",
        full_name: "Herbal-AI User",
        role: "user",
        is_active: true,
      });
      setTokens({
        access_token: "dev_token",
        refresh_token: "",
        expires_in: 3600,
        token_type: "bearer",
      });
      void (async () => {
        try {
          const { setTokenGetter } = await import("@/lib/api");
          setTokenGetter(async () => "dev_token");
          setTokenReady(true);
        } catch {}
      })();
      setIsLoading(false);
      return;
    }

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
  }, [isLoaded, userLoaded, isSignedIn, clerkUser, getToken]);

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

function FallbackAuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>({
    id: "dev_user_1",
    email: "dev@herbalai.com",
    full_name: "Herbal-AI User",
    role: "user",
    is_active: true,
  });

  const [tokens] = useState<AuthTokens | null>({
    access_token: "dev_token",
    refresh_token: "",
    expires_in: 3600,
    token_type: "bearer",
  });

  useEffect(() => {
    // Synchronously set token getter for dev mode
    void (async () => {
      try {
        const { setTokenGetter } = await import("@/lib/api");
        setTokenGetter(async () => "dev_token");
      } catch (error) {
        console.error("Failed to install dev token getter:", error);
      }
    })();
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        tokens,
        isLoading: false,
        isAuthenticated: !!user,
        tokenReady: true,
        logout: () => setUser(null),
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const clerkKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
  const isClerkConfigured =
    clerkKey &&
    clerkKey.startsWith("pk_") &&
    !clerkKey.includes("ZXhhbXBsZS") &&
    !clerkKey.includes("placeholder");

  if (!isClerkConfigured) {
    return <FallbackAuthProvider>{children}</FallbackAuthProvider>;
  }

  return (
    <ClerkProvider publishableKey={clerkKey}>
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
