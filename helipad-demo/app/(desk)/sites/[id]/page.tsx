import { notFound } from "next/navigation";
import { Inspection } from "@/components/inspection";
import { loadSite } from "@/lib/data";

export default async function SitePage({ params }: PageProps<"/sites/[id]">) {
  const site = await loadSite((await params).id);
  if (!site) notFound();
  return <Inspection site={site} />;
}

// Read per request so HELIPAD_DEMO_DATA can change without a rebuild; the read is a local file, so blocking is fine.
export const instant = false;
