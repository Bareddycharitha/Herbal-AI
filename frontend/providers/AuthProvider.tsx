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
    };

    const loadUser = async () => {
      try {
        const token = await getToken();
        if (token) {
          // Set the token getter for API interceptor
          const { setTokenGetter } = await import("@/lib/api");
          setTokenGetter(() => getToken());

          setTokens({
            access_token: token,
            refresh_token: "",
            expires_in: 0,
            token_type: "bearer",
          });

          // Call backend to get normalized user profile (triggers auto-creation if needed)
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/auth/me`, {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          
          });

          if (response.ok) {
            const userData = await response.json();
            setUser(userData);
          } else {
            // Fallback to Clerk data if backend call fails
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
          }
        } else {
          setUser(null);
          setTokens(null);
        }
      } catch (error) {
        console.error("Failed to load user profile:", error);
        // Fallback to Clerk data on error
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
        setTokens(null);
      } finally {
        setIsLoading(false);
      }
    };

    loadUser();
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
