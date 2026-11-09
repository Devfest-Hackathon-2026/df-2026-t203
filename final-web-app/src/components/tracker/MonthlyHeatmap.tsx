import { useMemo, useState } from 'react'
import { makeMonthly } from '../../data/mock'

const colors = ['#2a2620', '#6b3b1c', '#c4631a', '#f2a43a', '#ffd58a']
const mins = ['No activity', '~10 min', '~25 min', '~45 min', '1h+']
const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export default function MonthlyHeatmap() {
  const [offset, setOffset] = useState(0)
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null)
  const grid = useMemo(() => makeMonthly(offset), [offset])

  return (
    <div className="panel relative p-6">
      <div className="mb-4 flex items-center justify-between">
        <h4 className="font-display text-3xl glow-text">Activity</h4>
        <div className="flex gap-3 text-lg">
          <button className="hover:text-acid" onClick={() => setOffset((o) => o + 1)} aria-label="Earlier">‹</button>
          <button className="hover:text-acid disabled:opacity-30" disabled={offset === 0} onClick={() => setOffset((o) => o - 1)} aria-label="Later">›</button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="flex gap-[5px]">
          {grid.map((col, wi) => (
            <div key={wi} className="flex flex-col gap-[5px]">
              <div className="h-5 text-xs text-cream/60">
                {col[0].date.getDate() <= 7 ? months[col[0].date.getMonth()] : ''}
              </div>
              {col.map((c, di) => (
                <div
                  key={di}
                  onMouseEnter={(e) => {
                    const r = e.currentTarget.getBoundingClientRect()
                    const p = e.currentTarget.closest('.panel')!.getBoundingClientRect()
                    setTip({
                      x: r.left - p.left + r.width / 2,
                      y: r.top - p.top - 8,
                      text: `${mins[c.level]} on ${c.date.toDateString()}`,
                    })
                  }}
                  onMouseLeave={() => setTip(null)}
                  className="h-4 w-4 rounded-[5px] transition-transform hover:scale-125"
                  style={{ background: colors[c.level], border: c.level === 0 ? '1px solid #3a352b' : 'none' }}
                />
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 flex items-center justify-end gap-2 text-sm text-cream/70">
        less
        {colors.map((c) => (
          <span key={c} className="h-4 w-4 rounded-[5px]" style={{ background: c, border: '1px solid #3a352b' }} />
        ))}
        more
      </div>

      {tip && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-md bg-black px-3 py-1.5 text-xs text-cream"
          style={{ left: tip.x, top: tip.y }}
        >
          {tip.text}
        </div>
      )}
    </div>
  )
}
