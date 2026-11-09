import { Link } from 'react-router-dom'
import { USER } from '../data/mock'
import WeeklyChart from '../components/tracker/WeeklyChart'
import MonthlyHeatmap from '../components/tracker/MonthlyHeatmap'

export default function Home() {
  return (
    <div className="h-full overflow-y-auto">
      <section className="vignette relative flex h-[calc(100vh-4.5rem)] min-h-[520px] items-center justify-center overflow-hidden border-b border-edge">
        <video
          className="absolute inset-0 h-full w-full object-cover"
          src="/media/hero-video-web.mp4"
          autoPlay
          muted
          loop
          playsInline
        />
        <div className="absolute inset-0 bg-black/45" />
        <div className="relative z-10 text-center">
          <h1 className="font-display text-5xl leading-tight glow-text md:text-7xl">
            What are we grinding today,
            <br />
            <span className="font-script text-6xl text-amber md:text-8xl">{USER.name}</span>
            <span className="text-cream">?</span>
          </h1>
          <div className="mt-12 flex justify-center gap-10 text-3xl">
            <Link to="/training" className="glow-btn">Train</Link>
            <Link to="/analysis" className="glow-btn">Analyze</Link>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-8 py-16">
        <h2 className="stencil mb-10 text-center text-2xl text-cream/90">Tracker</h2>
        <div className="grid gap-10 lg:grid-cols-2">
          <div>
            <h3 className="mb-3 text-center font-display text-2xl text-cream/80">Weekly</h3>
            <WeeklyChart />
          </div>
          <div>
            <h3 className="mb-3 text-center font-display text-2xl text-cream/80">Monthly</h3>
            <MonthlyHeatmap />
          </div>
        </div>
      </section>
    </div>
  )
}
