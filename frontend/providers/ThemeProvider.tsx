"use client";
import { ThemeProvider as NextThemesProvider } from "next-themes";
export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  return <NextThemesProvider attribute="class" defaultTheme="herbal" themes={["herbal", "dark", "light", "midnight", "purple"]} enableSystem={false}>{children}</NextThemesProvider>;
}
