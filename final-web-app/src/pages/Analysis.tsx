import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import PageBackground from '../components/PageBackground'
import RigViewer from '../components/RigViewer'
import Sidebar from '../components/Sidebar'
import { mirroredPoseErrorRegion, poseErrorsAt, sustainedPoseErrors } from '../data/poseErrors'
import { useAnalysisPlayback, type Checkpoint } from '../data/useAnalysisPlayback'
import {
  outputUrl,
  post,
  recordingUrl,
  request,
  teacherUrl,
  upload,
  useLibrary,
  type Feedback,
  type Recording,
} from '../data/api'

function playbackCheckpoints(pairs: [number, number][], videoDuration: number, rigDuration: number) {
  const checkpoints: Checkpoint[] = []
  if (!videoDuration || !rigDuration) return checkpoints
  for (const [videoTime, rigTime] of pairs) {
    if (!Number.isFinite(videoTime) || videoTime < 0 || !Number.isFinite(rigTime) || rigTime < 0) continue
    // Allow a small difference between source and exported animation lengths.
    if (videoTime > videoDuration + 0.25 || rigTime > rigDuration + 0.25) continue
    const pose = { videoTime: Math.min(videoTime, videoDuration), rigTime: Math.min(rigTime, rigDuration) }
    const previous = checkpoints[checkpoints.length - 1]
    if (previous && (
      pose.rigTime < previous.rigTime || pose.videoTime < previous.videoTime ||
      (pose.rigTime === previous.rigTime && pose.videoTime === previous.videoTime)
    )) continue
    checkpoints.push(pose)
  }
  return checkpoints
}

function Review({
  recording,
  teacherFilename,
  refresh,
}: {
  recording: Recording
  teacherFilename?: string
  refresh: () => Promise<void>
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canCompare = Boolean(teacherFilename && recording.video_id !== 'misc_mocap')
  const [view, setView] = useState<'teacher' | 'clean'>(canCompare ? 'teacher' : 'clean')
  const comparing = view === 'teacher' && canCompare
  const videoSource = comparing && teacherFilename ? teacherUrl(teacherFilename) : recordingUrl(recording.filename)
  const rigSource = (recording.status === 'completed' || recording.is_processed) && recording.glb_filename
    ? outputUrl(recording, recording.glb_filename)
    : ''
  const [videoMetadata, setVideoMetadata] = useState({ source: '', duration: 0 })
  const [rigMetadata, setRigMetadata] = useState({ source: '', duration: 0 })
  const [cleanTimings, setCleanTimings] = useState<{ key: string; times: number[] } | null>(null)
  const [feedback, setFeedback] = useState<Feedback[] | 'unavailable' | null>(null)
  const [looping, setLooping] = useState(true)
  const [speed, setSpeed] = useState(1.0)
  const [stage, setStage] = useState('')
  const [logs, setLogs] = useState<string[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const videoDuration = videoMetadata.source === videoSource ? videoMetadata.duration : 0
  const rigLoaded = rigMetadata.source === rigSource
  const rigDuration = rigLoaded ? rigMetadata.duration : 0
  const duration = comparing ? rigDuration : videoDuration
  const base = `/api/recordings/${encodeURIComponent(recording.id)}`

  // The library polls periodically. Equivalent timing data must keep the current playback session.
  const matchData = useMemo(() => JSON.stringify(
    (recording.keypose_sync ?? []).map((pose) => [pose.teacher_time, pose.mocap_match_time]),
  ), [recording.keypose_sync])
  const cleanTimingKey = JSON.stringify([rigSource, recording.status, matchData])
  const cleanTimes = !comparing && cleanTimings?.key === cleanTimingKey ? cleanTimings.times : null
  const checkpoints = useMemo(() => playbackCheckpoints(
    comparing ? JSON.parse(matchData) : (cleanTimes ?? []).map((time) => [time, time]),
    videoDuration,
    rigDuration,
  ), [cleanTimes, comparing, matchData, rigDuration, videoDuration])
  const enabled = videoDuration > 0 && (
    comparing ? rigDuration > 0 : !rigSource || (rigLoaded && (!rigDuration || cleanTimes !== null))
  )
  const playback = useAnalysisPlayback({
    videoRef, source: videoSource, comparing, duration, videoDuration, checkpoints, enabled,
    speed, looping, onError: setError,
  })
  const { time, playing, status: keyposeStatus } = playback
  const noKeyposes = enabled && rigDuration > 0 && checkpoints.length === 0
  const errorTimeline = useMemo(() => Array.isArray(feedback) ? sustainedPoseErrors(feedback) : [], [feedback])
  const regionsAtTime = poseErrorsAt(errorTimeline, time)
  const activeErrors = recording.mocap_mirrored
    ? regionsAtTime.map((region) => mirroredPoseErrorRegion[region])
    : regionsAtTime

  useEffect(() => {
    if (!rigSource) return
    const controller = new AbortController()
    void request<Feedback[]>(`${base}/feedback`, { signal: controller.signal })
      .then((entries) => {
        if (!controller.signal.aborted) setFeedback(entries)
      })
      .catch(() => {
        if (!controller.signal.aborted) setFeedback('unavailable')
      })
    return () => controller.abort()
  }, [base, rigSource])

  useEffect(() => {
    if (!rigSource || recording.status === 'processing') return
    const controller = new AbortController()
    const storeTimes = (times: number[]) => {
      if (!controller.signal.aborted) setCleanTimings({ key: cleanTimingKey, times })
    }
    void request<number[]>(`${base}/keyposes`, { signal: controller.signal })
      .then(storeTimes)
      .catch(() => storeTimes([]))
    return () => controller.abort()
  }, [base, cleanTimingKey, recording.status, rigSource])

  useEffect(() => {
    if (recording.status !== 'processing') return
    const controller = new AbortController()
    let timer: number
    async function poll() {
      try {
        const status = await request<{ stage: string; logs?: string[]; status?: string }>(
          `${base}/status`,
          { signal: controller.signal },
        )
        if (!controller.signal.aborted) {
          setStage(status.stage)
          if (status.logs) setLogs(status.logs.slice(-6))
          if (status.status === 'completed' || status.status === 'error') {
            await refresh()
          }
        }
      } catch (err) {
        if (!controller.signal.aborted) setError((err as Error).message)
      }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 1500)
    }
    void poll()
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [base, recording.status, refresh])

  async function process() {
    setBusy(true)
    setError('')
    try {
      await post(`${base}/process`)
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const files = [
    ['Clean MP4', recordingUrl(recording.filename)],
    ['GLB', recording.glb_filename && outputUrl(recording, recording.glb_filename)],
    ['FBX', recording.fbx_filename && outputUrl(recording, recording.fbx_filename)],
    ['Blender', recording.blend_filename && outputUrl(recording, recording.blend_filename)],
  ] as const

  return (
    <div className="flex w-full max-w-6xl flex-1 flex-col justify-between gap-4">
      {/* Main 2-Column Split Panel (SOURCE VIDEO | 3D USER RIG) */}
      <div className="panel grid w-full flex-1 min-h-[380px] grid-cols-1 divide-y divide-edge overflow-hidden lg:grid-cols-2 lg:divide-x lg:divide-y-0">
        {/* Left Pane: SOURCE VIDEO */}
        <section className="relative h-full min-h-[300px] w-full overflow-hidden bg-black/55">
          <span className="stencil pointer-events-none absolute left-4 top-3 z-10 text-xs text-cream/60">
            SOURCE VIDEO · {comparing ? 'TEACHER REFERENCE' : 'CLEAN RECORDING'}
          </span>

          <video
            key={videoSource}
            ref={videoRef}
            src={videoSource}
            playsInline
            preload="auto"
            muted={comparing}
            className="absolute inset-0 h-full w-full object-contain"
            onLoadedMetadata={(event) => {
              const nextDuration = event.currentTarget.duration
              if (Number.isFinite(nextDuration)) setVideoMetadata({ source: videoSource, duration: nextDuration })
            }}
            onError={() => {
              playback.pause()
              setVideoMetadata({ source: videoSource, duration: 0 })
              setError(comparing ? 'The uploaded teacher video could not be loaded.' : 'The clean recording could not be loaded.')
              if (comparing) setView('clean')
            }}
          />

          {(keyposeStatus || noKeyposes) && (
            <div className="pointer-events-none absolute inset-x-3 bottom-3 z-10 border border-edge bg-black/80 px-3 py-1.5 text-xs text-amber-glow">
              {keyposeStatus || 'No key-pose timings available. Playing continuously.'}
            </div>
          )}
        </section>

        {/* Right Pane: 3D USER RIG */}
        <section className="relative h-full min-h-[300px] w-full overflow-hidden bg-black/55">
          <span className="stencil pointer-events-none absolute left-4 top-3 z-10 text-xs text-cream/60">
            3D USER RIG · {recording.title}
          </span>

          {rigSource ? (
            <RigViewer
              url={rigSource}
              mocapTime={time}
              poseErrors={activeErrors}
              onDurationLoaded={(dur) => setRigMetadata({ source: rigSource, duration: dur })}
            />
          ) : (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center gap-4 px-6 text-center"
              role="status"
            >
              <h2 className="font-display text-3xl text-cream/90">
                {recording.status === 'processing'
                  ? stage || 'Generating 3D user rig…'
                  : recording.error_message || '3D rig ready to generate'}
              </h2>
              {recording.status === 'processing' && logs.length > 0 && (
                <pre className="max-h-32 w-full max-w-md overflow-y-auto border border-edge bg-black/75 p-2.5 text-left text-xs text-cream/75">
                  {logs.join('\n')}
                </pre>
              )}
              {recording.status !== 'processing' && (
                <button
                  type="button"
                  className="glow-btn text-xl"
                  disabled={busy}
                  onClick={() => void process()}
                >
                  {recording.status === 'error' ? 'Retry 3D rig' : 'Generate 3D rig'}
                </button>
              )}
            </div>
          )}
        </section>
      </div>

      {rigDuration > 0 && feedback !== null && (
        <p className="text-center text-xs text-cream/70">
          {feedback === 'unavailable'
            ? 'Pose feedback could not be loaded.'
            : errorTimeline.length > 0
              ? <><span className="text-red-400">●</span> Red highlights mark sustained pose errors.{activeErrors.length > 0 && ` ${activeErrors.join(' · ')}`}</>
              : 'No sustained pose errors available for this recording.'}
        </p>
      )}

      {error && (
        <p role="alert" className="text-center text-sm text-red-300">
          {error}
        </p>
      )}

      {/* Bottom Controls Bar (All styled in the top menu bar's chamfered plate style) */}
      <div className="space-y-2.5">
        {/* Timeline Scrubber Row */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            className={`ctrl-btn ${playing ? 'active' : ''}`}
            disabled={!enabled}
            onClick={playback.toggle}
          >
            {playing ? 'Pause' : 'Play'}
          </button>

          <button
            type="button"
            className="ctrl-btn"
            disabled={!duration}
            onClick={() => playback.seek(0)}
          >
            Restart
          </button>

          {checkpoints.length > 0 && (
            <>
              <button
                type="button"
                className="ctrl-btn"
                onClick={() => playback.step(-1)}
                title="Previous Keyframe"
              >
                Prev Keyframe
              </button>
              <button
                type="button"
                className="ctrl-btn"
                onClick={() => playback.step(1)}
                title="Next Keyframe"
              >
                Next Keyframe
              </button>
            </>
          )}

          <button
            type="button"
            className={`ctrl-btn ${looping ? 'active' : ''}`}
            onClick={() => setLooping((l) => !l)}
          >
            Loop
          </button>

          <input
            aria-label="Recording timeline"
            list="analysis-keyframe-datalist"
            className="min-w-36 flex-1 accent-amber"
            type="range"
            min={0}
            max={duration || 1}
            step="0.01"
            value={time}
            disabled={!duration}
            onChange={(event) => playback.seek(Number(event.target.value), true)}
          />
          <datalist id="analysis-keyframe-datalist">
            {checkpoints.map((pose, i) => (
              <option key={i} value={pose.rigTime.toFixed(3)} label={`Keyframe ${i + 1}`} />
            ))}
          </datalist>

          <span className="stencil tabular-nums text-xs text-cream/80">
            {time.toFixed(1)}s / {duration.toFixed(1)}s
          </span>

          <label className="stencil flex items-center gap-1.5 border border-edge bg-ink/80 px-2.5 py-1 text-xs text-cream/80">
            Speed
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="bg-ink text-xs text-cream focus:outline-none"
            >
              <option value={0.5}>0.5x</option>
              <option value={1.0}>1.0x</option>
              <option value={1.5}>1.5x</option>
              <option value={2.0}>2.0x</option>
            </select>
          </label>
        </div>

        {/* View Toggle, Reprocess & Export Buttons Row */}
        <div className="flex flex-wrap items-center justify-center gap-2.5">
          {canCompare && (
            <>
              <button
                type="button"
                className={`ctrl-btn ${comparing ? 'active' : ''}`}
                onClick={() => {
                  playback.pause()
                  setView('teacher')
                }}
              >
                Teacher Video
              </button>
              <button
                type="button"
                className={`ctrl-btn ${!comparing ? 'active' : ''}`}
                onClick={() => {
                  playback.pause()
                  setView('clean')
                }}
              >
                Clean Recording
              </button>
            </>
          )}

          {recording.status !== 'processing' && (
            <button
              type="button"
              className="ctrl-btn"
              disabled={busy}
              onClick={() => void process()}
            >
              {rigSource ? 'Reprocess 3D Rig' : 'Generate 3D Rig'}
            </button>
          )}

          {files.map(
            ([label, url]) =>
              url && (
                <a key={label} href={url} download className="ctrl-btn">
                  Download {label}
                </a>
              ),
          )}
        </div>
      </div>
    </div>
  )
}

export default function Analysis() {
  const library = useLibrary()
  const [params, setParams] = useSearchParams()
  const id = params.get('recording')
  const recording = library.recordings.find((item) => item.id === id)
  const teacher = library.teachers.find((item) => item.id === recording?.video_id)
  const previousRecordingId = useRef<string | null>(null)

  useEffect(() => {
    if (id && previousRecordingId.current === id && !recording) {
      setParams({}, { replace: true })
    }
    previousRecordingId.current = recording?.id ?? null
  }, [id, recording, setParams])

  return (
    <PageBackground>
      <Sidebar
        {...library}
        selected={recording?.id ?? null}
        onRecording={(item) => setParams({ recording: item.id })}
        onDeleteRecording={library.deleteRecording}
        onUpload={async (file) => {
          const item = await upload<Recording>('/api/recordings/upload', file)
          setParams({ recording: item.id })
          await library.refresh()
        }}
      />
      <div className="flex min-w-0 flex-1 flex-col items-center justify-between gap-4 overflow-y-auto px-4 py-5 sm:px-10 sm:py-6">
        {library.error && (
          <p role="alert" className="text-sm text-red-300">
            {library.error}
          </p>
        )}

        {recording ? (
          <Review
            key={recording.id}
            recording={recording}
            teacherFilename={teacher?.filename ?? recording.teacher_video_filename}
            refresh={library.refresh}
          />
        ) : (
          <div className="flex w-full max-w-6xl flex-1 flex-col justify-between gap-4">
            <div className="panel grid w-full flex-1 min-h-[380px] grid-cols-1 divide-y divide-edge overflow-hidden lg:grid-cols-2 lg:divide-x lg:divide-y-0">
              <section className="relative flex h-full min-h-[300px] items-center justify-center px-6 text-center">
                <span className="stencil pointer-events-none absolute left-4 top-3 z-10 text-xs text-cream/60">
                  SOURCE VIDEO
                </span>
                <h2 className="font-display text-3xl text-cream/90">
                  {library.loading ? 'Loading recordings…' : 'Select a recording'}
                </h2>
              </section>
              <section className="relative flex h-full min-h-[300px] items-center justify-center px-6 text-center">
                <span className="stencil pointer-events-none absolute left-4 top-3 z-10 text-xs text-cream/60">
                  3D USER RIG
                </span>
                <h2 className="font-display text-3xl text-cream/90">
                  {library.recordings.length > 0
                    ? 'Select a recording to load 3D rig'
                    : 'Upload a recording to generate 3D rig'}
                </h2>
              </section>
            </div>

            {library.recordings.length > 0 && (
              <div className="flex justify-center">
                <button
                  type="button"
                  className="glow-btn text-xl"
                  onClick={() => setParams({ recording: library.recordings[0].id })}
                >
                  Open latest recording
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </PageBackground>
  )
}
