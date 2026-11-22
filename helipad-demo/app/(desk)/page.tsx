import { redirect } from "next/navigation";
import { loadIndex } from "@/lib/data";

export default async function Home() {
  const { sites } = await loadIndex();
  redirect(sites.length ? `/sites/${sites[0].id}` : "/new");
}
