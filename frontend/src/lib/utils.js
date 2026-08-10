import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges Tailwind class lists, resolving conflicts (e.g. "px-2 px-4" -> "px-4")
 * the way shadcn/ui components expect. Standard utility, present in every
 * shadcn-based project -- written by hand here since the shadcn CLI's
 * registry (ui.shadcn.com) isn't reachable from this sandbox's network
 * egress allowlist, but the underlying packages (clsx, tailwind-merge) are.
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
