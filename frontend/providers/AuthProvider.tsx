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
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

function AuthProviderInner({ children }: { children: ReactNode }) {
  const { isSignedIn, getToken, signOut } = useClerkAuth();
  const { user: clerkUser } = useClerkUser();
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!isSignedIn || !clerkUser) {
      setUser(null);
      setTokens(null);
      setIsLoading(false);
      return;
    }

    const loadToken = async () => {
      try {
        const token = await getToken();
        if (token) {
          setTokens({
            access_token: token,
            refresh_token: "",
            expires_in: 0,
            token_type: "bearer",
          });
        }
      } catch {
        // Token not available yet
      }

      setUser({
        id: clerkUser.id,
        email: clerkUser.emailAddresses[0]?.emailAddress ?? "",
        full_name:
          clerkUser.fullName ??
          clerkUser.firstName ??
          clerkUser.lastName ??
          null,
        role: "user",
        is_active: true,
      });
      setIsLoading(false);
    };

    loadToken();
  }, [isSignedIn, clerkUser, getToken]);

  async function logout() {
    await signOut();
    setUser(null);
    setTokens(null);
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        tokens,
        isLoading,
        isAuthenticated: !!user && !!tokens?.access_token,
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
