import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Sans_Hebrew } from "next/font/google";
import { cookies } from "next/headers";
import { LangProvider } from "@/components/lang";
import type { Lang } from "@/lib/i18n";
import "./globals.css";

const plex = IBM_Plex_Sans({ weight: ["400", "500", "600"], subsets: ["latin"], variable: "--font-plex" });
const plexHe = IBM_Plex_Sans_Hebrew({ weight: ["400", "500", "600"], subsets: ["hebrew"], variable: "--font-plex-he" });

export const metadata: Metadata = {
  title: "Landing zone inspection",
  description: "Helicopter landing zones checked against the inspection protocol.",
};

export default async function RootLayout({ children }: LayoutProps<"/">) {
  const lang: Lang = (await cookies()).get("lang")?.value === "he" ? "he" : "en";
  return (
    <html lang={lang} dir={lang === "he" ? "rtl" : "ltr"} className={`${plex.variable} ${plexHe.variable} h-full antialiased`}>
      <body className="h-full overflow-hidden">
        <LangProvider initial={lang}>{children}</LangProvider>
      </body>
    </html>
  );
}

// The language cookie and the recorded data are read per request; blocking routes are fine for a local desk.
export const instant = false;
