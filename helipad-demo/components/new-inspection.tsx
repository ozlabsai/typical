"use client";
// New inspection: a pin and a video. Live mode is not wired yet, so the form shows that plainly and produces no result.
import { MapPin } from "lucide-react";
import { useState } from "react";
import { useLang } from "@/components/lang";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function NewInspection() {
  const { d } = useLang();
  const [pin, setPin] = useState<[number, number] | null>(null);
  return (
    <div className="grid h-full place-items-center p-8">
      <form className="flex w-full max-w-[34rem] flex-col gap-6" onSubmit={(e) => e.preventDefault()}>
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">{d.ui.newTitle}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{d.ui.newLead}</p>
        </div>
        <fieldset className="flex flex-col gap-2">
          <legend className="mb-2 text-sm">{d.ui.pin}</legend>
          <button type="button" aria-label={d.ui.pin} dir="ltr"
            onClick={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              setPin([(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]);
            }}
            className="relative aspect-[16/7] w-full cursor-crosshair overflow-hidden rounded-lg border bg-card"
            style={{ backgroundImage: "repeating-linear-gradient(0deg, transparent 0 39px, #243034 39px 40px), repeating-linear-gradient(90deg, transparent 0 39px, #243034 39px 40px)" }}>
            {pin && (
              <MapPin aria-hidden className="absolute size-6 -translate-x-1/2 -translate-y-full text-foreground"
                style={{ left: `${pin[0] * 100}%`, top: `${pin[1] * 100}%` }} />
            )}
          </button>
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="lat">{d.ui.lat}</Label>
              <Input id="lat" inputMode="decimal" dir="ltr" className="num" />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="lon">{d.ui.lon}</Label>
              <Input id="lon" inputMode="decimal" dir="ltr" className="num" />
            </div>
          </div>
        </fieldset>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="video">{d.ui.videoFile}</Label>
          <Input id="video" type="file" accept="video/*" />
        </div>
        <div className="flex items-center gap-4">
          <Button type="submit" disabled aria-describedby="live-soon">{d.ui.start}</Button>
          <p id="live-soon" className="text-sm text-muted-foreground">{d.ui.liveSoon}</p>
        </div>
      </form>
    </div>
  );
}
