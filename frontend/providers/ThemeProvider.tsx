"use client";
import { ThemeProvider as NextThemesProvider } from "next-themes";

if (typeof window !== "undefined" && process.env.NODE_ENV === "development") {
  const origError = console.error;
  console.error = (...args: unknown[]) => {
    if (
      typeof args[0] === "string" &&
      args[0].includes("Encountered a script tag")
    ) {
      return;
    }
    origError.apply(console, args);
  };
}

export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  return <NextThemesProvider attribute="class" defaultTheme="herbal" themes={["herbal", "dark", "light", "midnight", "purple"]} enableSystem={false}>{children}</NextThemesProvider>;
}

