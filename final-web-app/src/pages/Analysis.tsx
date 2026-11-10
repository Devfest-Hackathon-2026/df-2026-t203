import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import PageBackground from '../components/PageBackground'
import RigViewer from '../components/RigViewer'
import Sidebar from '../components/Sidebar'
import {
  outputUrl,
  post,
  recordingUrl,
  request,
  teacherUrl,
  upload,
  useLibrary,
  type KeyposeSync,
  type Recording,
} from '../data/api'

function normalizeKeyposeSegments(
  rawSync: KeyposeSync[] | undefined,
  mocapDuration: number,
  teacherDuration: number,
): KeyposeSync[] {
  if (!rawSync || rawSync.length === 0) return []
  const dur = Math.max(1.0, mocapDuration || 15.0)
  const lastNextArrive = Number(rawSync[rawSync.length - 1].next_mocap_arrive_time || dur)
  const scale = lastNextArrive > 0.1 ? dur / lastNextArrive : 1.0

  return rawSync.map((item, idx) => {
    const arrive = Number(item.mocap_arrive_time || 0) * scale
    const nextArrive =
      idx + 1 === rawSync.length
        ? dur
        : Number(item.next_mocap_arrive_time || dur) * scale
    const tCur = Number(item.teacher_time || 0)
    let tNxt = Number(item.next_teacher_time ?? tCur)
    if (idx + 1 === rawSync.length && tNxt <= tCur) {
      tNxt = teacherDuration > tCur ? teacherDuration : tCur + 1.0
    }
    return {
      teacher_time: tCur,
      mocap_arrive_time: arrive,
      mocap_match_time: Number(item.mocap_match_time || arrive) * scale,
      next_teacher_time: tNxt,
      next_mocap_arrive_time: Math.max(arrive + 0.1, nextArrive),
    }
  })
}

function getSegmentIndexForTime(segments: KeyposeSync[], time: number): number {
  if (segments.length === 0) return 0
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i]
    if (
      time >= seg.mocap_arrive_time - 0.001 &&
      (time < seg.next_mocap_arrive_time - 0.001 || i === segments.length - 1)
    ) {
      return i
    }
  }
  return 0
}

function snapToNearestKeyframe(segments: KeyposeSync[], rawTime: number): number {
  if (segments.length === 0) return rawTime
  let best = segments[0].mocap_arrive_time
  let bestDist = Math.abs(rawTime - best)
  for (let i = 1; i < segments.length; i += 1) {
    const candidate = segments[i].mocap_arrive_time
    const dist = Math.abs(rawTime - candidate)
    if (dist < bestDist) {
      bestDist = dist
      best = candidate
    }
  }
  return best
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
  const cleanVideoRef = useRef<HTMLVideoElement>(null)
  const teacherVideoRef = useRef<HTMLVideoElement>(null)

  const canCompare = Boolean(teacherFilename && recording.video_id !== 'misc_mocap')
  const [view, setView] = useState<'teacher' | 'clean'>(canCompare ? 'teacher' : 'clean')
  const comparing = view === 'teacher' && canCompare

  const [time, setTime] = useState(0)
  const [cleanDuration, setCleanDuration] = useState(0)
  const [rigDuration, setRigDuration] = useState(0)
  const [teacherDuration, setTeacherDuration] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [looping, setLooping] = useState(true)
  const [speed, setSpeed] = useState(1.0)
  const [keyframeStatus, setKeyframeStatus] = useState('')
  const [activeKeyframeIdx, setActiveKeyframeIdx] = useState(0)

  const [stage, setStage] = useState('')
  const [logs, setLogs] = useState<string[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const duration = rigDuration || cleanDuration || 15.0
  const segments = useMemo(
    () => normalizeKeyposeSegments(recording.keypose_sync, duration, teacherDuration),
    [recording.keypose_sync, duration, teacherDuration],
  )

  const base = `/api/recordings/${encodeURIComponent(recording.id)}`

  // Mutable state for the 60fps mutual keyframe barrier loop
  const syncStateRef = useRef({
    time: 0,
    activeIdx: 0,
    bothReachedAt: 0,
    lastFrameTs: 0,
  })
  useEffect(() => {
    syncStateRef.current.time = time
  }, [time])

  useEffect(() => {
    setView(canCompare ? 'teacher' : 'clean')
    setTime(0)
    setPlaying(false)
    setActiveKeyframeIdx(0)
    syncStateRef.current = { time: 0, activeIdx: 0, bothReachedAt: 0, lastFrameTs: 0 }
  }, [recording.id, canCompare])

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

  // Seek helper for both videos and 3D mocap clock
  const seekToMocapTime = (nextTime: number, snapToKeypose = false) => {
    const clamped = Math.max(0, Math.min(duration, nextTime))
    const finalTime =
      comparing && snapToKeypose && segments.length > 0
        ? snapToNearestKeyframe(segments, clamped)
        : clamped

    setTime(finalTime)
    syncStateRef.current.time = finalTime
    syncStateRef.current.bothReachedAt = 0

    if (cleanVideoRef.current && Math.abs(cleanVideoRef.current.currentTime - finalTime) > 0.03) {
      try {
        cleanVideoRef.current.currentTime = finalTime
      } catch {}
    }

    if (comparing && segments.length > 0) {
      const idx = getSegmentIndexForTime(segments, finalTime)
      const seg = segments[idx]
      syncStateRef.current.activeIdx = idx
      setActiveKeyframeIdx(idx)
      if (teacherVideoRef.current && teacherVideoRef.current.readyState >= 1) {
        try {
          teacherVideoRef.current.currentTime = seg.teacher_time
        } catch {}
      }
      setKeyframeStatus(
        `⏸ Keyframe #${idx + 1}/${segments.length} (${seg.teacher_time.toFixed(1)}s)`,
      )
    }
  }

  const stepKeyframe = (delta: number) => {
    if (segments.length === 0) return
    const curIdx = syncStateRef.current.activeIdx
    const nextIdx = Math.max(0, Math.min(segments.length - 1, curIdx + delta))
    const targetSeg = segments[nextIdx]
    syncStateRef.current.activeIdx = nextIdx
    setActiveKeyframeIdx(nextIdx)
    seekToMocapTime(targetSeg.mocap_arrive_time, false)
  }

  // Mutual Keyframe Barrier Synchronization Loop
  useEffect(() => {
    if (!comparing) {
      setKeyframeStatus('')
      return
    }
    const teacherEl = teacherVideoRef.current
    if (!teacherEl || segments.length === 0) return

    if (!playing) {
      if (!teacherEl.paused) teacherEl.pause()
      const idx = getSegmentIndexForTime(segments, syncStateRef.current.time)
      const seg = segments[idx]
      setActiveKeyframeIdx(idx)
      setKeyframeStatus(
        `⏸ Keyframe #${idx + 1}/${segments.length} (${seg.teacher_time.toFixed(1)}s) — Paused at Keyframe`,
      )
      return
    }

    // Initialize segment when starting playback
    const startIdx = getSegmentIndexForTime(segments, syncStateRef.current.time)
    const startSeg = segments[startIdx]
    syncStateRef.current.activeIdx = startIdx
    syncStateRef.current.bothReachedAt = 0
    syncStateRef.current.lastFrameTs = performance.now()
    setActiveKeyframeIdx(startIdx)

    teacherEl.playbackRate = speed
    if (
      teacherEl.readyState >= 1 &&
      (teacherEl.currentTime < startSeg.teacher_time - 0.1 ||
        teacherEl.currentTime > startSeg.next_teacher_time + 0.1)
    ) {
      try {
        teacherEl.currentTime = startSeg.teacher_time
      } catch {}
    }
    if (startSeg.next_teacher_time > startSeg.teacher_time + 0.02 && teacherEl.paused) {
      void teacherEl.play().catch(() => undefined)
    }

    let raf = 0
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      const dt = Math.min(0.1, Math.max(0, (now - syncStateRef.current.lastFrameTs) / 1000))
      syncStateRef.current.lastFrameTs = now

      const total = segments.length
      const activeIdx = Math.max(0, Math.min(total - 1, syncStateRef.current.activeIdx))
      const seg = segments[activeIdx]
      const nextKeyframeNum = Math.min(total, activeIdx + 2)

      // Advance 3D mocap clock up to the current keyframe barrier (seg.next_mocap_arrive_time)
      const nextMocapTime = Math.min(
        seg.next_mocap_arrive_time,
        syncStateRef.current.time + dt * speed,
      )
      syncStateRef.current.time = nextMocapTime
      setTime(nextMocapTime)

      const mocapReached = nextMocapTime >= seg.next_mocap_arrive_time - 0.015
      const teacherSpan = Math.max(0, seg.next_teacher_time - seg.teacher_time)
      const videoUnavailable = Boolean(teacherEl.error || teacherEl.readyState < 1)
      const teacherReached =
        videoUnavailable ||
        teacherSpan <= 0.02 ||
        teacherEl.currentTime >= seg.next_teacher_time - 0.025 ||
        teacherEl.ended

      if (teacherReached) {
        if (!teacherEl.paused) teacherEl.pause()
      } else {
        if (Math.abs(teacherEl.playbackRate - speed) > 0.01) {
          teacherEl.playbackRate = speed
        }
        if (teacherEl.paused && !teacherEl.seeking) {
          void teacherEl.play().catch(() => undefined)
        }
      }

      if (teacherReached && !mocapReached) {
        syncStateRef.current.bothReachedAt = 0
        setKeyframeStatus(
          `⏸ Keyframe #${nextKeyframeNum}/${total} (${seg.next_teacher_time.toFixed(1)}s) — Video Waiting for 3D Mocap (${nextMocapTime.toFixed(1)}s / ${seg.next_mocap_arrive_time.toFixed(1)}s)`,
        )
        return
      }

      if (mocapReached && !teacherReached) {
        syncStateRef.current.bothReachedAt = 0
        setKeyframeStatus(
          `⏸ Keyframe #${nextKeyframeNum}/${total} — 3D Mocap Waiting for Video (${teacherEl.currentTime.toFixed(1)}s / ${seg.next_teacher_time.toFixed(1)}s)`,
        )
        return
      }

      if (!mocapReached && !teacherReached) {
        syncStateRef.current.bothReachedAt = 0
        setKeyframeStatus(
          `▶ Moving to Keyframe #${nextKeyframeNum}/${total} (Video ${teacherEl.currentTime.toFixed(1)}s/${seg.next_teacher_time.toFixed(1)}s • Mocap ${nextMocapTime.toFixed(1)}s/${seg.next_mocap_arrive_time.toFixed(1)}s)`,
        )
        return
      }

      // BOTH reached the next keypoint!
      if (!syncStateRef.current.bothReachedAt) {
        syncStateRef.current.bothReachedAt = now
      }
      setKeyframeStatus(`✓ Keyframe #${nextKeyframeNum}/${total} Reached by Both! Continuing...`)

      if (now - syncStateRef.current.bothReachedAt < 200) {
        return
      }

      if (activeIdx + 1 < total) {
        const nextIdx = activeIdx + 1
        const nextSeg = segments[nextIdx]
        syncStateRef.current.activeIdx = nextIdx
        syncStateRef.current.bothReachedAt = 0
        setActiveKeyframeIdx(nextIdx)

        teacherEl.playbackRate = speed
        if (Math.abs(teacherEl.currentTime - nextSeg.teacher_time) > 0.2 && !teacherEl.seeking) {
          try {
            teacherEl.currentTime = nextSeg.teacher_time
          } catch {}
        }
        if (nextSeg.next_teacher_time > nextSeg.teacher_time + 0.02) {
          void teacherEl.play().catch(() => undefined)
        }
      } else if (looping) {
        const firstSeg = segments[0]
        syncStateRef.current.activeIdx = 0
        syncStateRef.current.time = firstSeg.mocap_arrive_time
        syncStateRef.current.bothReachedAt = 0
        setActiveKeyframeIdx(0)
        setTime(firstSeg.mocap_arrive_time)
        try {
          teacherEl.currentTime = firstSeg.teacher_time
        } catch {}
        teacherEl.playbackRate = speed
        if (firstSeg.next_teacher_time > firstSeg.teacher_time + 0.02) {
          void teacherEl.play().catch(() => undefined)
        }
      } else {
        if (!teacherEl.paused) teacherEl.pause()
        setPlaying(false)
        setKeyframeStatus(`✓ All ${total} Keyframes Completed`)
      }
    }

    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      teacherEl.pause()
    }
  }, [comparing, playing, segments, speed, looping])

  // Clean recording playback sync when not in teacher-comparison mode
  useEffect(() => {
    if (comparing) return
    const cleanEl = cleanVideoRef.current
    if (!cleanEl) return
    cleanEl.playbackRate = speed
    cleanEl.loop = looping
    if (playing) {
      if (cleanEl.paused) void cleanEl.play().catch(() => undefined)
    } else if (!cleanEl.paused) {
      cleanEl.pause()
    }
  }, [comparing, playing, speed, looping])

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

  const togglePlay = () => {
    if (comparing) {
      setPlaying((prev) => !prev)
    } else if (cleanVideoRef.current) {
      if (cleanVideoRef.current.paused) {
        void cleanVideoRef.current
          .play()
          .then(() => setPlaying(true))
          .catch(() => setError('Playback could not start.'))
      } else {
        cleanVideoRef.current.pause()
        setPlaying(false)
      }
    } else {
      setPlaying((prev) => !prev)
    }
  }

  const files = [
    ['Clean MP4', recordingUrl(recording.filename)],
    ['GLB', recording.glb_filename && outputUrl(recording, recording.glb_filename)],
    ['FBX', recording.fbx_filename && outputUrl(recording, recording.fbx_filename)],
    ['Blender', recording.blend_filename && outputUrl(recording, recording.blend_filename)],
  ] as const

  const hasRig = (recording.status === 'completed' || recording.is_processed) && Boolean(recording.glb_filename)

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
            ref={cleanVideoRef}
            src={recordingUrl(recording.filename)}
            playsInline
            preload="metadata"
            muted={comparing}
            className={`absolute inset-0 h-full w-full object-contain ${
              comparing ? 'invisible' : ''
            }`}
            onLoadedMetadata={(event) => setCleanDuration(event.currentTarget.duration)}
            onTimeUpdate={(event) => {
              if (!comparing) setTime(event.currentTarget.currentTime)
            }}
            onPlay={() => {
              if (!comparing) setPlaying(true)
            }}
            onPause={() => {
              if (!comparing) setPlaying(false)
            }}
            onEnded={() => {
              if (!comparing) setPlaying(false)
            }}
            onError={() => setError('The clean recording could not be loaded.')}
          />

          {comparing && teacherFilename && (
            <video
              ref={teacherVideoRef}
              src={teacherUrl(teacherFilename)}
              muted
              playsInline
              preload="auto"
              className="absolute inset-0 h-full w-full object-contain"
              onLoadedMetadata={(event) => setTeacherDuration(event.currentTarget.duration)}
              onError={() => {
                setError('The uploaded teacher video could not be loaded.')
                setView('clean')
              }}
            />
          )}

          {comparing && keyframeStatus && (
            <div className="pointer-events-none absolute inset-x-3 bottom-3 z-10 border border-edge bg-black/80 px-3 py-1.5 text-xs text-amber-glow">
              {keyframeStatus}
            </div>
          )}
        </section>

        {/* Right Pane: 3D USER RIG */}
        <section className="relative h-full min-h-[300px] w-full overflow-hidden bg-black/55">
          <span className="stencil pointer-events-none absolute left-4 top-3 z-10 text-xs text-cream/60">
            3D USER RIG · {recording.title}
          </span>

          {hasRig && recording.glb_filename ? (
            <RigViewer
              url={outputUrl(recording, recording.glb_filename)}
              video={cleanVideoRef}
              mocapTime={time}
              onDurationLoaded={(dur) => setRigDuration(dur)}
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
            disabled={!duration}
            onClick={togglePlay}
          >
            {playing ? 'Pause' : 'Play'}
          </button>

          <button
            type="button"
            className="ctrl-btn"
            disabled={!duration}
            onClick={() => {
              setPlaying(false)
              seekToMocapTime(0, false)
            }}
          >
            Restart
          </button>

          {comparing && segments.length > 0 && (
            <>
              <button
                type="button"
                className="ctrl-btn"
                onClick={() => stepKeyframe(-1)}
                title="Previous Keyframe"
              >
                Prev Keyframe
              </button>
              <button
                type="button"
                className="ctrl-btn"
                onClick={() => stepKeyframe(1)}
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
            onChange={(event) => {
              const next = Number(event.target.value)
              if (playing) setPlaying(false)
              seekToMocapTime(next, true)
            }}
          />
          <datalist id="analysis-keyframe-datalist">
            {segments.map((seg, i) => (
              <option key={i} value={seg.mocap_arrive_time.toFixed(3)} label={`Keyframe ${i + 1}`} />
            ))}
          </datalist>

          <span className="stencil tabular-nums text-xs text-cream/80">
            {time.toFixed(1)}s / {duration.toFixed(1)}s
            {comparing && segments.length > 0 && ` · KF ${activeKeyframeIdx + 1}/${segments.length}`}
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
                  setPlaying(false)
                  setView('teacher')
                }}
              >
                Teacher Video
              </button>
              <button
                type="button"
                className={`ctrl-btn ${!comparing ? 'active' : ''}`}
                onClick={() => {
                  setPlaying(false)
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
              {hasRig ? 'Reprocess 3D Rig' : 'Generate 3D Rig'}
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

  return (
    <PageBackground>
      <Sidebar
        {...library}
        selected={recording?.id ?? null}
        onRecording={(item) => setParams({ recording: item.id })}
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
