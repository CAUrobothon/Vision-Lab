// What the server can do, from GET /api/health `features` (route groups: folders, gt, score). The
// app shell reads it once and every screen asks here, so a server started before a feature existed
// simply shows none of its parts (no requests to routes it lacks, no errors).
import { createContext, useContext } from 'react'

export type Feature = 'folders' | 'gt' | 'score'

/** The server's features; null until /api/health has answered. */
export const FeaturesContext = createContext<ReadonlySet<string> | null>(null)

export function useFeatures(): ReadonlySet<string> | null {
  return useContext(FeaturesContext)
}

/** True once the server has said it has `f` (false while it has not answered yet). */
export function useFeature(f: Feature): boolean {
  return useContext(FeaturesContext)?.has(f) ?? false
}
