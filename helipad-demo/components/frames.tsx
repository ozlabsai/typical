"use client";
// Video keyframes and the model's answers: one keyframe at a time (tabs by timestamp), every option of every
// question with its probability. During the replay the tabs follow the frame being read.
import { motion } from "motion/react";
import { useState } from "react";
import { useLang } from "@/components/lang";
import { neutral, ProbBar, ProbNumbers } from "@/components/prob-bar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ltr, tr } from "@/lib/i18n";
import { NULL_OPTION, ordered, QUESTION_OPTIONS, type Frame, type Question } from "@/lib/protocol";

export function Frames({ frames, done, active }: { frames: Frame[]; done: number; active: number }) {
  const { d } = useLang();
  const [picked, setPicked] = useState(0);
  const shown = active >= 0 ? active : Math.min(picked, Math.max(0, done - 1));
  const time = (f: Frame) => `${ltr(f.t.toFixed(1))} ${d.ui.seconds}`;
  return (
    <Tabs value={String(shown)} onValueChange={(v) => setPicked(Number(v))} className="gap-2" aria-label={d.ui.video}>
      <div className="flex items-center gap-3">
        <h2 className="text-xs text-muted-foreground">{d.ui.video}</h2>
        <TabsList variant="line" className="h-7">
          {frames.map((f, i) => (
            <TabsTrigger key={f.image} value={String(i)} disabled={i >= done} className="px-2 text-xs">
              {time(f)}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>
      {frames.map((f, i) => (
        <TabsContent key={f.image} value={String(i)} className="flex h-48 flex-none gap-4">
          {/* eslint-disable-next-line @next/next/no-img-element -- served from the data dir by /data */}
          <img src={`/data/${f.image}`} alt={`${d.ui.frameAt} ${time(f)}`}
            className="h-full w-[6.75rem] shrink-0 rounded-md border object-cover" />
          <div className="thin-scroll min-w-0 flex-1 columns-2 gap-x-5 overflow-y-auto pe-1">
            {i < done && Object.entries(f.answers).map(([q, probs], j) => {
              const entries = ordered(probs, [...(QUESTION_OPTIONS[q as Question] ?? []), NULL_OPTION]);
              const top = entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
              return (
                <motion.div key={q} className="mb-2.5 break-inside-avoid" initial={active === i ? { opacity: 0 } : false} animate={{ opacity: 1 }}
                  transition={{ duration: 0.25, delay: active === i ? j * 0.06 : 0 }}>
                  <div className="mb-1 text-xs">{tr(d.questions, q)}</div>
                  <ProbBar size="sm" entries={entries} names={d.options} label={tr(d.questions, q)} color={neutral} />
                  <ProbNumbers className="mt-1" entries={entries} names={d.options} top={top} color={neutral} />
                </motion.div>
              );
            })}
          </div>
        </TabsContent>
      ))}
    </Tabs>
  );
}
