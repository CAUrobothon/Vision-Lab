// Which overlay layers are visible. Shared by every viewer and remembered in localStorage.
import { useSyncExternalStore } from 'react'
import { sharedValue } from '../lib/storage'

export interface LayerPrefs {
  quad: boolean // 윗면 사각형
  center: boolean // 중심 · 각도
  label: boolean // 라벨 (색 + 각도)
  raw: boolean // 원 출력 (polygon / box)
  roi: boolean // 검출 영역
  gt: boolean // 정답 (the media's ground truth, when it has one)
  debug: boolean // 디버그 레이어 (asks the preview for debug images)
  debugName: string | null // chosen debug layer; null = the first one
  debugOpacity: number // 0..1
  debugBlend: 'normal' | 'screen' // colour debug images: plain, or only their bright parts
}

const DEFAULTS: LayerPrefs = {
  quad: true,
  center: true,
  label: true,
  raw: false,
  roi: true,
  gt: false,
  debug: false,
  debugName: null,
  debugOpacity: 0.6,
  debugBlend: 'normal',
}

const store = sharedValue<LayerPrefs>('layers', DEFAULTS, (raw) => ({ ...DEFAULTS, ...(raw as Partial<LayerPrefs> | null) }))

/** Turn layers on or off for every viewer (and remember it). */
export function setLayerPrefs(patch: Partial<LayerPrefs>) {
  store.set({ ...store.get(), ...patch })
}

export function useLayerPrefs(): [LayerPrefs, (patch: Partial<LayerPrefs>) => void] {
  return [useSyncExternalStore(store.subscribe, store.get), setLayerPrefs]
}
