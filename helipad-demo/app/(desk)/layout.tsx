import { Desk } from "@/components/desk";
import { loadIndex } from "@/lib/data";

export default async function DeskLayout({ children }: LayoutProps<"/">) {
  return <Desk index={await loadIndex()}>{children}</Desk>;
}

export const instant = false;
