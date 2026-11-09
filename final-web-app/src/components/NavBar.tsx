import { useEffect, useRef, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { USER } from '../data/mock'

const tabs = [
  { to: '/', label: 'Home' },
  { to: '/training', label: 'Training' },
  { to: '/analysis', label: 'Analysis' },
]

export default function NavBar() {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  return (
    <header className="relative z-30 grid grid-cols-[1fr_auto_1fr] items-center border-b border-edge bg-ink/90 px-8 py-3 backdrop-blur">
      <NavLink to="/" className="font-display text-3xl text-cream glow-text">
        Motion<span className="text-amber">Learn</span>
      </NavLink>

      <nav className="flex items-center gap-10">
        {tabs.map((t) => (
          <NavLink
            key={t.to}
            to={t.to}
            end
            className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}
          >
            {t.label}
          </NavLink>
        ))}
      </nav>

      <div className="relative justify-self-end" ref={ref}>
        <button
          onClick={() => setOpen((o) => !o)}
          aria-label="Profile"
          className="grid h-11 w-11 place-items-center rounded-full border border-edge bg-concrete/60 text-cream transition hover:border-acid hover:text-acid"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <circle cx="12" cy="8" r="4" />
            <path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7" />
          </svg>
        </button>
        {open && (
          <div className="panel absolute right-0 top-14 w-56 py-2 shadow-2xl">
            <div className="border-b border-edge px-4 pb-2">
              <div className="font-display text-xl">{USER.name}</div>
              <div className="stencil text-[10px] text-cream/50">Main Menu</div>
            </div>
            {['Profile', 'Settings', 'Sign out'].map((l) => (
              <button
                key={l}
                className="block w-full px-4 py-2 text-left text-sm text-cream/80 hover:bg-white/5 hover:text-acid"
              >
                {l}
              </button>
            ))}
          </div>
        )}
      </div>
    </header>
  )
}
