"use client";
import { MotionConfig } from "motion/react";
import { Direction } from "radix-ui";
import { createContext, useContext, useState } from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { DICT, type Dict, type Lang } from "@/lib/i18n";

const Ctx = createContext<{ lang: Lang; d: Dict; setLang: (l: Lang) => void } | null>(null);

export function LangProvider({ initial, children }: { initial: Lang; children: React.ReactNode }) {
  const [lang, set] = useState<Lang>(initial);
  const setLang = (l: Lang) => {
    set(l);
    document.cookie = `lang=${l}; path=/; max-age=31536000; samesite=lax`;
    document.documentElement.lang = l;
    document.documentElement.dir = l === "he" ? "rtl" : "ltr";
  };
  const dir = lang === "he" ? "rtl" : "ltr";
  return (
    <Ctx.Provider value={{ lang, d: DICT[lang], setLang }}>
      <Direction.Provider dir={dir}>
        <MotionConfig reducedMotion="user">{children}</MotionConfig>
      </Direction.Provider>
    </Ctx.Provider>
  );
}

export function useLang() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useLang outside LangProvider");
  return c;
}

export function LangToggle() {
  const { lang, d, setLang } = useLang();
  return (
    <ToggleGroup type="single" size="sm" variant="outline" spacing={0} value={lang} aria-label={d.ui.language}
      onValueChange={(v) => v && setLang(v as Lang)}>
      <ToggleGroupItem value="en" lang="en" className="px-3">EN</ToggleGroupItem>
      <ToggleGroupItem value="he" lang="he" className="px-3">עב</ToggleGroupItem>
    </ToggleGroup>
  );
}
