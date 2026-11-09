export const USER = { name: 'Saim' }

export const weekly = {
  label: 'Nov 9 – 16',
  days: [
    { day: 'Mon', min: 10 },
    { day: 'Tue', min: 50 },
    { day: 'Wed', min: 60 },
    { day: 'Thu', min: 60 },
    { day: 'Fri', min: 15 },
    { day: 'Sat', min: 0 },
    { day: 'Sun', min: 20 },
  ],
}

/** Deterministic pseudo-random heatmap: 7 rows x 26 weeks, levels 0-4 */
export function makeMonthly(offset = 0) {
  const weeks = 26
  const out: { date: Date; level: number }[][] = []
  const start = new Date(2025, 11, 1 - offset * 7 * weeks)
  let seed = 7 + offset * 13
  const rnd = () => ((seed = (seed * 9301 + 49297) % 233280) / 233280)
  for (let w = 0; w < weeks; w++) {
    const col = []
    for (let d = 0; d < 7; d++) {
      const date = new Date(start)
      date.setDate(start.getDate() + w * 7 + d)
      const r = rnd()
      col.push({ date, level: r < 0.3 ? 0 : r < 0.5 ? 1 : r < 0.7 ? 2 : r < 0.88 ? 3 : 4 })
    }
    out.push(col)
  }
  return out
}
