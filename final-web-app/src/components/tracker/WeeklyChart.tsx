import { useState } from 'react'
import { weekly } from '../../data/mock'

const fmt = (m: number) => (m === 0 ? '0 min' : m >= 60 ? `${m / 60}h` : `${m} min`)

export default function WeeklyChart() {
  const [hover, setHover] = useState<number | null>(null)
  const max = Math.max(...weekly.days.map((d) => d.min), 1)
  return (
    <div className="panel p-6">
      <div className="mb-6 flex items-center justify-between">
        <h4 className="font-display text-3xl glow-text">Weekly goal</h4>
        <div className="flex items-center gap-3 text-sm text-cream/70">
          <button className="hover:text-acid" aria-label="Previous week">‹</button>
          <span>{weekly.label}</span>
          <button className="hover:text-acid" aria-label="Next week">›</button>
        </div>
      </div>
      <div className="flex h-56 items-end justify-between gap-3">
        {weekly.days.map((d, i) => {
          const h = Math.max((d.min / max) * 100, d.min === 0 ? 0 : 14)
          return (
            <div
              key={d.day}
              className="flex h-full flex-1 flex-col items-center justify-end gap-2"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              <span className="text-xs text-cream/60">{fmt(d.min)}</span>
              <div className="flex w-full flex-1 items-end">
                <div
                  className="w-full rounded-full transition-all duration-300"
                  style={{
                    height: d.min === 0 ? 8 : `${h}%`,
                    background: d.min === 0 ? '#3a352b' : 'linear-gradient(#f2a43a,#c4631a)',
                    boxShadow: hover === i && d.min ? '0 0 22px rgba(242,164,58,.6)' : 'none',
                  }}
                />
              </div>
              <span className={`text-sm ${hover === i ? 'text-acid' : 'text-cream/80'}`}>{d.day}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
