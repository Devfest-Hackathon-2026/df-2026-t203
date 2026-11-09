import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import PageBackground from '../components/PageBackground'
import RigViewer from '../components/RigViewer'
import Sidebar from '../components/Sidebar'
import { outputUrl, post, recordingUrl, request, teacherUrl, upload, useLibrary, type Feedback, type KeyposeSync, type Recording } from '../data/api'

function teacherClock(entries: KeyposeSync[], time: number) {
  const entry = entries.find((item) => time < item.next_mocap_arrive_time) ?? entries[entries.length - 1]
  const window = entry.next_mocap_arrive_time - entry.mocap_match_time
  const delta = entry.next_teacher_time - entry.teacher_time
  if (time <= entry.mocap_match_time || window <= 0 || delta <= 0) return { time: entry.teacher_time, rate: 0 }
  const fraction = Math.min(1, (time - entry.mocap_match_time) / window)
  return { time: entry.teacher_time + fraction * delta, rate: fraction < 1 ? delta / window : 0 }
}

function Review({ recording, teacherFilename, refresh }: {
  recording: Recording
  teacherFilename?: string
  refresh: () => Promise<void>
}) {
  const video = useRef<HTMLVideoElement>(null)
  const teacher = useRef<HTMLVideoElement>(null)
  const [view, setView] = useState<'clean' | 'teacher'>('clean')
  const [feedback, setFeedback] = useState<Feedback[]>([])
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [stage, setStage] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const entries = recording.keypose_sync
  const syncEntries = useRef(entries)
  useEffect(() => { syncEntries.current = entries }, [entries])
  const canCompare = !!(teacherFilename && entries?.length)
  const comparing = view === 'teacher' && canCompare
  const base = `/api/recordings/${encodeURIComponent(recording.id)}`

  useEffect(() => {
    const controller = new AbortController()
    request<Feedback[]>(`${base}/feedback`, { signal: controller.signal }).then(setFeedback).catch((err: Error) => {
      if (!controller.signal.aborted) setError(err.message)
    })
    return () => controller.abort()
  }, [base])

  useEffect(() => {
    if (recording.status !== 'processing') return
    const controller = new AbortController()
    let timer: number
    async function poll() {
      try {
        const status = await request<{ stage: string }>(`${base}/status`, { signal: controller.signal })
        if (!controller.signal.aborted) setStage(status.stage)
      } catch (err) {
        if (!controller.signal.aborted) setError((err as Error).message)
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 2000)
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [base, recording.status])

  useEffect(() => {
    const source = video.current
    const reference = teacher.current
    if (!comparing || !source || !reference) return
    let raf: number
    let playPending = false
    const sync = () => {
      if (reference.readyState < 1 || !syncEntries.current?.length) return
      const target = teacherClock(syncEntries.current, source.currentTime)
      const shouldPlay = !source.paused && !source.seeking && target.rate >= 0.25 && target.rate <= 4
      if (!shouldPlay && !reference.paused) reference.pause()
      if (Math.abs(reference.currentTime - target.time) > (shouldPlay ? 0.15 : 0.025) && !reference.seeking) {
        reference.currentTime = target.time
      }
      if (shouldPlay) {
        reference.playbackRate = target.rate
        if (reference.paused && !playPending) {
          playPending = true
          void reference.play().catch(() => undefined).finally(() => { playPending = false })
        }
      }
    }
    const loop = () => { sync(); raf = requestAnimationFrame(loop) }
    loop()
    return () => { cancelAnimationFrame(raf); reference.pause() }
  }, [comparing])

  async function process() {
    setBusy(true)
    setError('')
    try { await post(`${base}/process`); await refresh() }
    catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }

  const files = [
    ['Clean MP4', recordingUrl(recording.filename)],
    ['GLB', recording.glb_filename && outputUrl(recording, recording.glb_filename)],
    ['FBX', recording.fbx_filename && outputUrl(recording, recording.fbx_filename)],
    ['Blender', recording.blend_filename && outputUrl(recording, recording.blend_filename)],
  ]

  return (
    <div className="w-full max-w-6xl space-y-3">
      <h1 className="font-display text-2xl">{recording.title}</h1>
      <div className="grid gap-2 lg:grid-cols-2">
        <section className="panel relative aspect-video overflow-hidden bg-black">
          <video ref={video} src={recordingUrl(recording.filename)} playsInline preload="metadata" muted={comparing}
            className={`h-full w-full object-contain ${comparing ? 'invisible absolute inset-0' : ''}`}
            onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
            onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
            onError={() => setError('The clean recording could not be loaded.')} />
          {comparing && <video ref={teacher} src={teacherUrl(teacherFilename!)} muted playsInline className="h-full w-full object-contain"
            onError={() => { setError('The teacher video could not be loaded.'); setView('clean') }} />}
          <span className="stencil absolute left-2 top-2 bg-black/75 px-2 py-1 text-xs">{comparing ? 'Teacher video' : 'Clean recording'}</span>
        </section>
        <section className="panel relative aspect-video overflow-hidden">
          <span className="pointer-events-none stencil absolute left-2 top-2 z-10 text-xs">3D user rig</span>
          {recording.status === 'completed' && recording.glb_filename ? (
            <RigViewer url={outputUrl(recording, recording.glb_filename)} video={video} feedback={feedback} />
          ) : (
            <div className="grid h-full place-content-center gap-2 px-4 text-center text-sm" role="status">
              <p>{recording.status === 'processing' ? stage || 'Generating the 3D rig…' : recording.error_message || 'The 3D rig is ready to generate.'}</p>
              {recording.status !== 'processing' && <button className="text-acid underline" disabled={busy} onClick={() => void process()}>
                {recording.status === 'error' ? 'Retry 3D processing' : 'Generate 3D rig'}
              </button>}
            </div>
          )}
        </section>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <button className="border border-edge px-3 py-1 text-acid disabled:opacity-50" disabled={!duration}
          onClick={() => {
            if (video.current?.paused) void video.current.play().catch(() => setError('Playback could not start.'))
            else video.current?.pause()
          }}>{playing ? 'Pause' : 'Play'}</button>
        <input aria-label="Recording timeline" className="min-w-24 flex-1" type="range" min={0} max={duration || 1} step="0.01"
          value={time} disabled={!duration} onChange={(event) => {
            const next = Number(event.target.value)
            if (video.current) video.current.currentTime = next
            setTime(next)
          }} />
        <span>{time.toFixed(1)} / {duration.toFixed(1)} s</span>
        {canCompare && <button className="text-acid underline" onClick={() => setView(comparing ? 'clean' : 'teacher')}>
          {comparing ? 'Show clean recording' : 'Compare with teacher'}
        </button>}
      </div>
      {feedback.length > 0 && <p className="text-sm text-cream/75"><span className="mr-2 text-red-400">●</span>Red markers show joint-angle errors during keypose holds.</p>}
      {error && <p role="alert" className="text-red-300">{error}</p>}
      <div className="flex flex-wrap gap-4 text-sm text-acid">
        {files.map(([label, url]) => url && <a key={label} href={url} download className="underline">Download {label}</a>)}
      </div>
    </div>
  )
}

export default function Analysis() {
  const library = useLibrary()
  const [params, setParams] = useSearchParams()
  const id = params.get('recording')
  const recording = library.recordings.find((item) => item.id === id) ?? (!id ? library.recordings[0] : undefined)
  const teacher = library.teachers.find((item) => item.id === recording?.video_id)

  return (
    <PageBackground>
      <Sidebar {...library} selected={recording?.id ?? null} onRecording={(item) => setParams({ recording: item.id })}
        onUpload={async (file) => {
          const item = await upload<Recording>('/api/recordings/upload', file)
          setParams({ recording: item.id })
          await library.refresh()
        }} />
      <div className="flex min-w-0 flex-1 flex-col items-center gap-3 overflow-y-auto p-3 sm:p-8">
        {library.error && <p role="alert" className="text-red-300">{library.error}</p>}
        {recording ? <Review key={recording.id} recording={recording} teacherFilename={teacher?.filename ?? recording.teacher_video_filename} refresh={library.refresh} />
          : <p>{library.loading ? 'Loading recordings…' : id ? 'Recording unavailable.' : 'Select a recording or upload a video to analyze.'}</p>}
      </div>
    </PageBackground>
  )
}
