// Korean particles after a name the user typed: "DS2"를, "옵션A"와, "HSV 색 범위"로.
// The particle depends on whether the name's last sound ends in a consonant (and, for 으로/로,
// whether that consonant is ㄹ). Hangul is exact; digits and Latin letters use their Korean reading.

type Final = 'none' | 'rieul' | 'other'

const DIGIT: Record<string, Final> = {
  '0': 'other', '1': 'rieul', '2': 'none', '3': 'other', '4': 'none',
  '5': 'none', '6': 'other', '7': 'rieul', '8': 'rieul', '9': 'none',
}
// Letters as read aloud (엘, 엠, 엔, 알 end in a consonant; 에이, 비, 씨 ... do not).
const LETTER: Record<string, Final> = { l: 'rieul', r: 'rieul', m: 'other', n: 'other' }

function finalOf(word: string): Final {
  const chars = [...word.trim()].filter((c) => !/[\s"'`)\]}>.,!?·-]/.test(c))
  const ch = chars[chars.length - 1]
  if (!ch) return 'none'
  const code = ch.codePointAt(0)!
  if (code >= 0xac00 && code <= 0xd7a3) {
    const jong = (code - 0xac00) % 28
    return jong === 0 ? 'none' : jong === 8 ? 'rieul' : 'other'
  }
  if (ch in DIGIT) return DIGIT[ch]
  return LETTER[ch.toLowerCase()] ?? 'none'
}

export type JosaPair = '을/를' | '이/가' | '은/는' | '과/와' | '으로/로' | '이에요/예요'

/** The particle that follows `word`. */
export function josa(word: string, pair: JosaPair): string {
  const [withFinal, without] = pair.split('/')
  const f = finalOf(word)
  if (pair === '으로/로') return f === 'other' ? withFinal : without
  return f === 'none' ? without : withFinal
}

/** `"name"` followed by its particle: named('DS2', '을/를') -> "DS2"를 */
export function named(name: string, pair: JosaPair): string {
  return `"${name}"${josa(name, pair)}`
}
