// Live pipeline preview: POST /api/preview for every change of the request, latest wins.
//
// At most one request is in flight. The server keeps computing a request even after the browser
// aborts it, so aborting on every slider step piles work up on large images (nothing ever
// finishes). Instead a change while a request runs only remembers the newest body, and that one
// is sent the moment the running request settles; the bodies in between are never sent. On small
// images a request takes tens of milliseconds, so the preview still follows a drag closely.
import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { PreviewRequest, PreviewResult } from '../api/types'

export interface PreviewState {
  result: PreviewResult | null
  error: string | null
  pending: boolean
  rtt: number | null // last round trip in ms (browser side)
}

const IDLE: PreviewState = { result: null, error: null, pending: false, rtt: null }

export function usePreview(req: PreviewRequest | null): PreviewState {
  const key = req ? JSON.stringify(req) : null
  const [state, setState] = useState<PreviewState>(IDLE)
  // inFlight: the running request; next: the newest body asked for while it runs.
  const ctl = useRef({ inFlight: null as AbortController | null, next: null as string | null })

  useEffect(() => {
    const c = ctl.current
    const send = (body: string) => {
      const ac = new AbortController()
      c.inFlight = ac
      c.next = null
      const start = performance.now()
      const settle = (patch: (s: PreviewState) => PreviewState) => {
        if (ac.signal.aborted) return
        c.inFlight = null
        const queued = c.next !== body ? c.next : null
        c.next = null
        // Stay "pending" while a newer body waits, so the badge does not flicker during a drag.
        setState((s) => ({ ...patch(s), pending: queued !== null }))
        if (queued !== null) send(queued)
      }
      api.preview(JSON.parse(body) as PreviewRequest, ac.signal).then(
        (result) => settle(() => ({ result, error: null, pending: false, rtt: performance.now() - start })),
        (err: Error) => {
          if (err.name !== 'AbortError') settle((s) => ({ ...s, error: err.message }))
        },
      )
    }

    if (!key) {
      // Nothing to preview (a run is shown, or the video plays): drop the queue and the request.
      c.next = null
      c.inFlight?.abort()
      c.inFlight = null
      setState((s) => (s.pending ? { ...s, pending: false } : s))
      return
    }
    setState((s) => (s.pending ? s : { ...s, pending: true }))
    if (c.inFlight) c.next = key
    else send(key)
  }, [key])

  useEffect(() => {
    const c = ctl.current
    return () => {
      c.next = null
      c.inFlight?.abort()
      c.inFlight = null
    }
  }, [])

  return state
}
