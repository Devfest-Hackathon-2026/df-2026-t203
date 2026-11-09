import { useState } from 'react'
import type { Recording, Teacher } from '../data/api'

interface Props {
  teachers: Teacher[]
  recordings: Recording[]
  selected: string | null
  onTeacher?: (teacher: Teacher) => void
  onRecording: (recording: Recording) => void
  onUpload: (file: File) => Promise<void>
  onYouTube?: (url: string) => Promise<void>
  disabled?: boolean
  loading: boolean
}

export default function Sidebar({
  teachers, recordings, selected, onTeacher, onRecording, onUpload, onYouTube, disabled, loading,
}: Props) {
  const [collapsed, setCollapsed] = useState(() => window.matchMedia('(max-width: 639px)').matches)
  const [youtube, setYoutube] = useState(false)
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const submit = async (action: () => Promise<void>) => {
    setBusy(true)
    setError('')
    try { await action(); setYoutube(false); setUrl('') }
    catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }

  const renderRecording = (recording: Recording) => (
    <li key={recording.id}>
      <button disabled={disabled} onClick={() => onRecording(recording)}
        aria-current={selected === recording.id ? 'true' : undefined}
        className="w-full px-3 py-1.5 text-left text-sm hover:text-acid aria-[current=true]:text-brush disabled:opacity-50">
        {recording.title}
        <span className="ml-2 text-xs text-cream/60">{recording.status}</span>
      </button>
    </li>
  )
  const standalone = recordings.filter((recording) => !teachers.some((teacher) => teacher.id === recording.video_id))

  return (
    <aside className={`panel relative z-20 flex shrink-0 flex-col border-y-0 border-l-0 ${collapsed ? 'w-12' : 'w-48 sm:w-72'}`}>
      <button onClick={() => setCollapsed(!collapsed)} aria-label="Toggle library" aria-expanded={!collapsed}
        className="m-2 self-start border border-edge px-2 py-1">
        {collapsed ? '☰' : '‹ Library'}
      </button>
      {!collapsed && (
        <>
          <div className="space-y-2 border-b border-edge px-3 py-2 text-sm">
            <label className="block">
              {onYouTube ? 'Upload teacher video' : 'Upload a recording'}
              <input type="file" accept="video/*,.mp4,.mov,.webm" disabled={busy || disabled}
                className="mt-1 block w-full text-xs"
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0]
                  event.currentTarget.value = ''
                  if (file) void submit(() => onUpload(file))
                }} />
            </label>
            {onYouTube && (
              <button className="text-acid underline disabled:opacity-50" disabled={busy || disabled} onClick={() => setYoutube(!youtube)}>
                Paste YouTube link
              </button>
            )}
            {youtube && onYouTube && (
              <form onSubmit={(event) => { event.preventDefault(); if (url.trim()) void submit(() => onYouTube(url.trim())) }}>
                <input type="url" required aria-label="YouTube URL" value={url} onChange={(event) => setUrl(event.target.value)}
                  disabled={busy || disabled} className="w-full border border-edge bg-ink px-2 py-1" placeholder="https://youtu.be/…" />
                <button className="mt-1 text-acid" disabled={busy || disabled}>Import</button>
              </form>
            )}
            {busy && <p role="status">Importing video…</p>}
            {error && <p role="alert" className="text-red-300">{error}</p>}
          </div>
          <nav aria-label="Media library" className="flex-1 overflow-y-auto px-3 pb-4">
            {loading && <p className="py-2 text-sm">Loading library…</p>}
            {!loading && !teachers.length && !recordings.length && <p className="py-2 text-sm">The library is empty.</p>}
            {teachers.map((teacher) => {
              const takes = recordings.filter((recording) => recording.video_id === teacher.id)
              if (!onTeacher && !takes.length) return null
              return (
                <section key={teacher.id} className="mt-3">
                  {onTeacher ? (
                    <button onClick={() => onTeacher(teacher)} disabled={disabled}
                      aria-current={selected === teacher.id ? 'true' : undefined}
                      className="w-full text-left font-display text-xl hover:text-acid aria-[current=true]:text-brush disabled:opacity-50">
                      {teacher.name} <span className="text-xs text-cream/60">{teacher.status}</span>
                    </button>
                  ) : <h2 className="font-display text-xl">{teacher.name}</h2>}
                  <ul className="border-l border-edge">{takes.map(renderRecording)}</ul>
                </section>
              )
            })}
            {standalone.length > 0 && (
              <section className="mt-3">
                <h2 className="font-display text-xl">Standalone recordings</h2>
                <ul className="border-l border-edge">{standalone.map(renderRecording)}</ul>
              </section>
            )}
          </nav>
        </>
      )}
    </aside>
  )
}
