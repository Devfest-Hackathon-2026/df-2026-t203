import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PageBackground from '../components/PageBackground'
import Sidebar from '../components/Sidebar'
import { post, teacherUrl, upload, useLibrary, type Teacher } from '../data/api'
import { useTraining } from '../data/useTraining'

const button = 'border border-edge bg-ink px-3 py-1.5 text-sm hover:text-acid disabled:opacity-50'

export default function Training() {
  const library = useLibrary()
  const navigate = useNavigate()
  const [selected, setSelected] = useState<string | null>(null)
  const [opacity, setOpacity] = useState(50)
  const [mirror, setMirror] = useState(true)
  const [videoReady, setVideoReady] = useState('')
  const [duration, setDuration] = useState(0)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const teacher = library.teachers.find((item) => item.id === selected) ?? library.teachers[0]
  const training = useTraining((recording) => navigate(`/analysis?recording=${encodeURIComponent(recording.id)}`))
  const active = training.phase !== 'IDLE'
  const feedback = training.feedback
  const [width, height] = training.dimensions

  async function selectImported(promise: Promise<Teacher>) {
    const video = await promise
    setSelected(video.id)
    await library.refresh()
  }

  async function update(path: string, body?: object) {
    setBusy(true)
    setError('')
    try { await post(path, body); await library.refresh() }
    catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }

  return (
    <PageBackground>
      <Sidebar {...library} selected={teacher?.id ?? null} disabled={active}
        onTeacher={(item) => { setSelected(item.id); setError('') }}
        onRecording={(item) => navigate(`/analysis?recording=${encodeURIComponent(item.id)}`)}
        onUpload={(file) => selectImported(upload<Teacher>('/api/videos/upload', file))}
        onYouTube={(url) => selectImported(post<Teacher>('/api/videos/youtube?auto_process=true', { url }))} />
      <div className="flex min-w-0 flex-1 flex-col items-center gap-4 overflow-y-auto p-3 sm:p-8">
        <div className="panel relative aspect-video w-full max-w-5xl overflow-hidden bg-black">
          {teacher ? (
            <video key={teacher.id} ref={training.teacher} src={teacherUrl(teacher.filename)} playsInline preload="metadata"
              controls={!active} muted className="absolute inset-0 h-full w-full object-cover"
              onLoadedMetadata={(event) => { setVideoReady(teacher.id); setDuration(event.currentTarget.duration) }}
              onError={() => { setVideoReady(''); setError('The teacher video could not be loaded.') }} />
          ) : <p className="absolute inset-0 grid place-items-center">Upload or select a teacher video</p>}
          <div className={`pointer-events-none absolute inset-0 ${mirror ? '-scale-x-100' : ''}`}>
            <video ref={training.camera} autoPlay muted playsInline className="h-full w-full object-cover"
              style={{ opacity: active ? opacity / 100 : 0 }} />
            {feedback?.arrows.length ? (
              <svg aria-hidden="true" className="absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid slice">
                <defs><marker id="correction-tip" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
                  <path d="M0,0 L5,2.5 L0,5 Z" fill="#ff5555" />
                </marker></defs>
                {feedback.arrows.map(({ start, end }, index) => (
                  <line key={index} x1={start[0] * width} y1={start[1] * height} x2={end[0] * width} y2={end[1] * height}
                    stroke="#ff5555" strokeWidth={4} markerEnd="url(#correction-tip)" />
                ))}
              </svg>
            ) : null}
          </div>
          {active && (
            <div className="pointer-events-none absolute inset-x-2 top-2 flex flex-wrap gap-2 text-xs sm:text-sm">
              <span className="bg-black/75 px-2 py-1 text-red-300">{training.phase === 'SAVING' ? 'Saving clean take…' : training.phase === 'STARTING' ? 'Connecting camera…' : '● Training'}</span>
              {feedback && (
                <span className="bg-black/75 px-2 py-1">
                  Pose {feedback.pose}/{feedback.total}
                  {training.phase === 'PAUSED' && ` · Hold ${feedback.hold}/${feedback.required_hold} · Error ${feedback.score?.toFixed(1) ?? '–'}°`}
                  {feedback.dtw != null && ` · Movement score ${feedback.dtw.toFixed(0)}%`}
                </span>
              )}
            </div>
          )}
        </div>
        {(training.error || error || library.error) && <p role="alert" className="text-red-300">{training.error || error || library.error}</p>}
        {teacher && !teacher.is_processed && (
          <div className="flex items-center gap-3" role="status">
            <span>{teacher.status === 'processing' ? 'Extracting keyposes…' : teacher.error_message || 'Extract keyposes to start training.'}</span>
            {teacher.status !== 'processing' && <button className={button} disabled={busy}
              onClick={() => void update(`/api/videos/${teacher.id}/process`)}>{teacher.status === 'error' ? 'Retry extraction' : 'Extract keyposes'}</button>}
          </div>
        )}
        {teacher?.is_processed && !active && (
          <form key={`${teacher.id}-${teacher.checkpoint_interval}`} className="flex flex-wrap items-end gap-2"
            onSubmit={(event) => {
              event.preventDefault()
              const seconds = Number(new FormData(event.currentTarget).get('interval'))
              void update(`/api/videos/${teacher.id}/checkpoint`, { checkpoint_interval: seconds || null })
            }}>
            <label className="flex flex-col text-sm">
              Checkpoint interval (seconds)
              <input name="interval" type="number" min={0} max={duration || undefined} step="any" defaultValue={teacher.checkpoint_interval ?? 0}
                className="w-32 border border-edge bg-ink px-2 py-1" />
            </label>
            <button className={button} disabled={busy}>Apply</button>
            <span className="text-sm text-cream/70">0 uses the full clip · {teacher.keypose_count} keyposes</span>
          </form>
        )}
        <div className="flex flex-wrap items-center justify-center gap-3">
          {active ? (
            <>
              <button className={button} disabled={training.phase === 'SAVING'} onClick={() => training.control('STOP_SESSION')}>Stop and save</button>
              <button className={button} disabled={!['PLAYING', 'PAUSED', 'CHECKPOINT'].includes(training.phase)}
                onClick={() => training.control('PREVIOUS_CHECKPOINT')}>Previous checkpoint</button>
              {training.phase === 'PAUSED' && <button className={button} onClick={() => training.control('SKIP_POSE')}>Skip pose</button>}
              {training.phase === 'CHECKPOINT' && (
                <>
                  <button className={button} onClick={() => training.control('CHECKPOINT_ACTION', 'Thumb_Up')}>Continue</button>
                  <button className={button} onClick={() => training.control('CHECKPOINT_ACTION', 'Thumb_Down')}>Retry checkpoint</button>
                </>
              )}
            </>
          ) : (
            <button className="glow-btn text-2xl disabled:opacity-40" disabled={!teacher?.is_processed || videoReady !== teacher.id || busy}
              onClick={() => teacher && void training.start(teacher.id)}>Start training</button>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-center gap-4 text-sm">
          <label className="flex items-center gap-2">Camera opacity
            <input type="range" min={0} max={100} value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} />
            {opacity}%
          </label>
          <label><input type="checkbox" checked={mirror} onChange={(event) => setMirror(event.target.checked)} /> Mirror camera</label>
        </div>
        {training.phase === 'CHECKPOINT' && <p className="text-sm text-cream/80">Continue with a thumbs up or retry with a thumbs down. A rewind replaces footage from the selected checkpoint.</p>}
      </div>
    </PageBackground>
  )
}
