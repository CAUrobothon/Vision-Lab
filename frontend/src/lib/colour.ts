// Pixel colour conversions that reproduce what the backend computes, so the inspector shows
// the same numbers the pipeline parameters use:
// - HSV on OpenCV's 8-bit scale (cv2.COLOR_BGR2HSV): H 0–179, S 0–255, V 0–255.
// - Lab in real units like cubevision.colour.bgr_to_lab (OpenCV's float path, sRGB, D65).
// HSV ports OpenCV's fixed-point RGB2HSV_b, because the float formula is off by one on ~2% of
// pixels. Checked against cv2 on 5000 random pixels: HSV identical except 2 pixels off by 1 in S;
// Lab within 0.35 units of bgr_to_lab (mean 0.08): cv2 interpolates its float Lab conversion
// through a lookup table, this is the exact formula.

export type Rgb = [number, number, number]

const HSV_SHIFT = 12
const SDIV = new Int32Array(256)
const HDIV = new Int32Array(256)
for (let i = 1; i < 256; i++) {
  SDIV[i] = Math.round((255 << HSV_SHIFT) / i)
  HDIV[i] = Math.round((180 << HSV_SHIFT) / (6 * i))
}

export function rgbToHsvCv([r, g, b]: Rgb): Rgb {
  const v = Math.max(r, g, b)
  const diff = v - Math.min(r, g, b)
  const half = 1 << (HSV_SHIFT - 1)
  const s = (diff * SDIV[v] + half) >> HSV_SHIFT
  let h = v === r ? g - b : v === g ? b - r + 2 * diff : r - g + 4 * diff
  h = (h * HDIV[diff] + half) >> HSV_SHIFT
  if (h < 0) h += 180
  return [h, s, v]
}

function srgbToLinear(c: number): number {
  const x = c / 255
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}

function labF(t: number): number {
  return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116
}

/** Lab in real units (L 0–100, a/b about −128..127), as cubevision.colour.bgr_to_lab returns it
 *  (OpenCV's float path: sRGB gamma, D65, the same constants as cv2). */
export function rgbToLabCv(rgb: Rgb): Rgb {
  const [r, g, b] = rgb.map(srgbToLinear)
  const x = (0.412453 * r + 0.35758 * g + 0.180423 * b) / 0.950456
  const y = 0.212671 * r + 0.71516 * g + 0.072169 * b
  const z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754
  const fx = labF(x)
  const fy = labF(y)
  const fz = labF(z)
  const L = y > 0.008856 ? 116 * fy - 16 : 903.3 * y
  return [L, 500 * (fx - fy), 200 * (fy - fz)]
}

export function rgbToHex([r, g, b]: Rgb): string {
  return '#' + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')
}

export function hexToRgb(hex: string): Rgb | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) return null
  const n = parseInt(m[1], 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** Per-channel median of an RGBA patch (ImageData.data), ignoring alpha. */
export function medianRgb(data: Uint8ClampedArray): Rgb {
  const chans: number[][] = [[], [], []]
  for (let i = 0; i < data.length; i += 4) {
    chans[0].push(data[i])
    chans[1].push(data[i + 1])
    chans[2].push(data[i + 2])
  }
  const med = (xs: number[]) => xs.sort((p, q) => p - q)[Math.floor(xs.length / 2)]
  return [med(chans[0]), med(chans[1]), med(chans[2])]
}
