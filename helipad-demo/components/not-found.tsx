"use client";
import Link from "next/link";
import { useLang } from "@/components/lang";

export function NotFound() {
  const { d } = useLang();
  return (
    <div className="grid h-full place-content-center gap-3 text-center">
      <p>{d.ui.notFound}</p>
      <Link href="/" className="text-sm text-muted-foreground underline underline-offset-4">{d.ui.back}</Link>
    </div>
  );
}
