import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...inputs: (string | undefined | false | null)[]) =>
  twMerge(clsx(inputs));
