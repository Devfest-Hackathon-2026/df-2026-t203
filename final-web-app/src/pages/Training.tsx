import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PageBackground from '../components/PageBackground'
import Sidebar from '../components/Sidebar'
import { post, teacherUrl, upload, useLibrary, type Teacher } from '../data/api'
import { useTraining } from '../data/useTraining'

type CameraMode = 'pip' | 'overlay' | 'sideBySide'

export default function Training() {
  const library = useLibrary()
  const navigate = useNavigate()
  const [selected, setSelected] = useState<string | null>(null)
  const [cameraMode, setCameraMode] = useState<CameraMode>('pip')
  const [opacity, setOpacity] = useState(50)
  const [mirror, setMirror] = useState(true)
  const [videoReady, setVideoReady] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const teacher = library.teachers.find((item) => item.id === selected) ?? null
  const training = useTraining((recording) =>
    navigate(`/analysis?recording=${encodeURIComponent(recording.id)}`),
  )
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
    try {
      await post(path, body)
      await library.refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const handleStartRecording = () => {
    const targetTeacher = teacher ?? library.teachers[0]
    if (!targetTeacher) {
      setError('Select or upload a teacher video from the left library first.')
      return
    }
    if (!selected) {
      setSelected(targetTeacher.id)
    }
    if (!targetTeacher.is_processed) {
      void update(`/api/videos/${targetTeacher.id}/process`)
      return
    }
    void training.start(targetTeacher.id)
  }

  return (
    <PageBackground>
      <Sidebar
        {...library}
        selected={teacher?.id ?? null}
        disabled={active}
        onTeacher={(item) => {
          setSelected(item.id)
          setError('')
        }}
        onRecording={(item) => navigate(`/analysis?recording=${encodeURIComponent(item.id)}`)}
        onDeleteRecording={library.deleteRecording}
        onUpload={(file) => selectImported(upload<Teacher>('/api/videos/upload', file))}
        onYouTube={(url) =>
          selectImported(post<Teacher>('/api/videos/youtube?auto_process=true', { url }))
        }
      />

      <div className="flex min-w-0 flex-1 flex-col items-center justify-between gap-4 overflow-y-auto px-4 py-5 sm:px-10 sm:py-6">
        {/* Main Stage Frame */}
        <div
          className={`panel relative w-full max-w-5xl flex-1 min-h-[380px] overflow-hidden ${
            cameraMode === 'sideBySide'
              ? 'grid grid-cols-1 divide-y divide-edge lg:grid-cols-2 lg:divide-x lg:divide-y-0'
              : ''
          }`}
        >
          {/* Reference Teacher Video Pane */}
          <div className="relative h-full min-h-[300px] w-full overflow-hidden">
            {teacher ? (
              <>
                <span className="stencil pointer-events-none absolute left-4 top-3 z-10 text-xs text-cream/65">
                  REFERENCE VIDEO · {teacher.name}
                </span>
                <video
                  key={teacher.id}
                  ref={training.teacher}
                  src={teacherUrl(teacher.filename)}
                  playsInline
                  preload="metadata"
                  controls={!active}
                  muted
                  className="absolute inset-0 h-full w-full object-contain bg-black/40"
                  onLoadedMetadata={() => setVideoReady(teacher.id)}
                  onError={() => {
                    setVideoReady('')
                    setError('The teacher video could not be loaded.')
                  }}
                />
              </>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center px-6 text-center">
                <h2 className="font-display text-3xl text-cream">Select a teacher video</h2>
                <p className="stencil mt-1.5 text-xs text-cream/60">REFERENCE VIDEO</p>
              </div>
            )}

            {active && (
              <div className="pointer-events-none absolute inset-x-4 top-10 z-20 flex flex-wrap gap-2 text-xs sm:text-sm">
                <span className="border border-edge bg-black/80 px-2.5 py-1 text-amber">
                  {training.phase === 'SAVING'
                    ? 'Saving clean take…'
                    : training.phase === 'STARTING'
                      ? 'Connecting camera…'
                      : '● Recording in progress'}
                </span>
                {feedback && (
                  <span className="border border-edge bg-black/80 px-2.5 py-1 text-cream">
                    Pose {feedback.pose}/{feedback.total}
                    {training.phase === 'PAUSED' &&
                      ` · Hold ${feedback.hold}/${feedback.required_hold} · Error ${feedback.score?.toFixed(1) ?? '–'}°`}
                    {feedback.dtw != null && ` · Movement score ${feedback.dtw.toFixed(0)}%`}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* User Camera Pane (PiP corner box, Full Translucent Overlay, or Side-by-Side) */}
          <div
            className={
              cameraMode === 'pip'
                ? 'absolute bottom-4 right-4 z-20 aspect-video w-48 overflow-hidden border border-amber-deep/65 bg-black shadow-2xl sm:w-64'
                : cameraMode === 'overlay'
                  ? 'pointer-events-none absolute inset-0 z-10 overflow-hidden'
                  : 'relative h-full min-h-[300px] w-full overflow-hidden bg-black/60'
            }
          >
            {cameraMode === 'sideBySide' && (
              <span className="stencil pointer-events-none absolute left-4 top-3 z-10 text-xs text-cream/65">
                YOUR CAMERA
              </span>
            )}
            <div className={`pointer-events-none absolute inset-0 ${mirror ? '-scale-x-100' : ''}`}>
              <video
                ref={training.camera}
                autoPlay
                muted
                playsInline
                className="h-full w-full object-cover"
                style={{
                  opacity: active ? (cameraMode === 'overlay' ? opacity / 100 : 1) : 0,
                }}
              />
              {feedback?.arrows.length ? (
                <svg
                  aria-hidden="true"
                  className="absolute inset-0 h-full w-full"
                  viewBox={`0 0 ${width} ${height}`}
                  preserveAspectRatio="xMidYMid slice"
                >
                  <defs>
                    <marker
                      id="correction-tip"
                      markerWidth="5"
                      markerHeight="5"
                      refX="4"
                      refY="2.5"
                      orient="auto"
                    >
                      <path d="M0,0 L5,2.5 L0,5 Z" fill="#ff5555" />
                    </marker>
                  </defs>
                  {feedback.arrows.map(({ start, end }, index) => (
                    <line
                      key={index}
                      x1={start[0] * width}
                      y1={start[1] * height}
                      x2={end[0] * width}
                      y2={end[1] * height}
                      stroke="#ff5555"
                      strokeWidth={4}
                      markerEnd="url(#correction-tip)"
                    />
                  ))}
                </svg>
              ) : null}
            </div>
          </div>
        </div>

        {/* Status / Error Notifications */}
        {(training.error || error || library.error) && (
          <p role="alert" className="text-sm text-red-300">
            {training.error || error || library.error}
          </p>
        )}

        {/* Bottom Controls Bar (Top-Menu Chamfered Button Style) */}
        <div className="flex w-full max-w-5xl flex-col items-center gap-3">
          {/* Primary Action Row */}
          <div className="flex flex-wrap items-center justify-center gap-3">
            {active ? (
              <>
                <button
                  type="button"
                  className="glow-btn text-xl"
                  disabled={training.phase === 'SAVING'}
                  onClick={() => training.control('STOP_SESSION')}
                >
                  Stop and save
                </button>
                {training.phase === 'PAUSED' && (
                  <button
                    type="button"
                    className="ctrl-btn"
                    onClick={() => training.control('SKIP_POSE')}
                  >
                    Skip pose
                  </button>
                )}
              </>
            ) : (
              <button
                type="button"
                className="glow-btn text-2xl"
                disabled={
                  busy ||
                  (teacher != null &&
                    teacher.is_processed &&
                    videoReady !== teacher.id) ||
                  teacher?.status === 'processing'
                }
                onClick={handleStartRecording}
              >
                {teacher && !teacher.is_processed
                  ? teacher.status === 'processing'
                    ? 'Extracting keyposes…'
                    : 'Extract keyposes'
                  : 'Start recording'}
              </button>
            )}
          </div>

          {/* Secondary Controls Row (View Mode, Mirror, Opacity, Keyposes) */}
          <div className="flex flex-wrap items-center justify-center gap-2.5">
            <button
              type="button"
              className={`ctrl-btn ${cameraMode === 'pip' ? 'active' : ''}`}
              onClick={() => setCameraMode('pip')}
            >
              PiP View
            </button>
            <button
              type="button"
              className={`ctrl-btn ${cameraMode === 'overlay' ? 'active' : ''}`}
              onClick={() => setCameraMode('overlay')}
            >
              Overlay View
            </button>
            <button
              type="button"
              className={`ctrl-btn ${cameraMode === 'sideBySide' ? 'active' : ''}`}
              onClick={() => setCameraMode('sideBySide')}
            >
              Side-by-Side
            </button>
            <button
              type="button"
              className={`ctrl-btn ${mirror ? 'active' : ''}`}
              onClick={() => setMirror((m) => !m)}
            >
              Mirror Camera
            </button>

            {cameraMode === 'overlay' && (
              <label className="stencil flex items-center gap-2 border border-edge bg-ink/80 px-3 py-1 text-xs text-cream/85">
                Opacity
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={opacity}
                  onChange={(event) => setOpacity(Number(event.target.value))}
                  className="w-20 accent-amber"
                />
                <span>{opacity}%</span>
              </label>
            )}

            {teacher && !active && !teacher.is_processed && teacher.status !== 'processing' && (
              <button
                type="button"
                className="ctrl-btn"
                disabled={busy}
                onClick={() => void update(`/api/videos/${teacher.id}/process`)}
              >
                {teacher.status === 'error' ? 'Retry Keyposes' : 'Extract Keyposes'}
              </button>
            )}
          </div>
        </div>
      </div>
    </PageBackground>
  )
}
