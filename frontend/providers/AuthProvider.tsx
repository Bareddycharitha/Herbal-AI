"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react";
import {
  login as apiLogin,
  register as apiRegister,
  logout as apiLogout,
  getAccessToken,
  isAuthenticated,
  getApiBaseUrl,
} from "@/lib/api";
import type { AuthTokens, User, LoginCredentials, RegisterCredentials } from "@/types";

interface AuthContextType {
  user: User | null;
  tokens: AuthTokens | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (credentials: RegisterCredentials) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is already logged in on mount
    const token = getAccessToken();
    if (token) {
      // Token exists — try to fetch user info
      fetchUserProfile(token).then((profile) => {
        setUser(profile);
        setTokens({
          access_token: token,
          refresh_token: "",
          expires_in: 0,
          token_type: "bearer",
        });
      }).catch(() => {
        // Token is invalid — clear it
        apiLogout();
      }).finally(() => {
        setIsLoading(false);
      });
    } else {
      setIsLoading(false);
    }
  }, []);

  async function fetchUserProfile(token: string): Promise<User> {
    const response = await fetch(`${getApiBaseUrl()}/api/v1/auth/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error("Failed to fetch user profile");
    }

    return response.json();
  }

  async function login(credentials: LoginCredentials) {
    const data = await apiLogin(credentials.email, credentials.password);
    setTokens(data);

    // Fetch user profile after login
    const profile = await fetchUserProfile(data.access_token);
    setUser(profile);
  }

  async function register(credentials: RegisterCredentials) {
    await apiRegister(credentials.email, credentials.password, credentials.full_name);
  }

  function logout() {
    apiLogout();
    setUser(null);
    setTokens(null);
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        tokens,
        isLoading,
        isAuthenticated: !!user && !!getAccessToken(),
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
