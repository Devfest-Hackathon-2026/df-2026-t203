import { useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react'

export interface Checkpoint {
  videoTime: number
  rigTime: number
}

const HOLD_MS = 1000
const TIME_EPSILON = 0.001

function teacherTimeAt(checkpoints: Checkpoint[], rigTime: number, rigDuration: number, videoDuration: number) {
  let previous = { rigTime: 0, videoTime: 0 }
  const end = { rigTime: rigDuration, videoTime: videoDuration }
  for (const next of [...checkpoints, end]) {
    if (rigTime <= next.rigTime) {
      const span = next.rigTime - previous.rigTime
      return span > 0
        ? previous.videoTime + ((rigTime - previous.rigTime) / span) * (next.videoTime - previous.videoTime)
        : next.videoTime
    }
    previous = next
  }
  return videoDuration
}

export function useAnalysisPlayback({
  videoRef, source, comparing, videoDuration, duration, checkpoints, enabled, speed, looping, onError,
}: {
  videoRef: RefObject<HTMLVideoElement>
  source: string
  comparing: boolean
  videoDuration: number
  duration: number
  checkpoints: Checkpoint[]
  enabled: boolean
  speed: number
  looping: boolean
  onError: (message: string) => void
}) {
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [status, setStatus] = useState('')
  const session = useRef({
    time: 0,
    running: false,
    next: 0,
    videoAtPose: false,
    holdElapsed: 0,
    waitingForStart: true,
    finished: false,
  }).current
  const options = useRef({ speed, looping })
  useLayoutEffect(() => {
    options.current = { speed, looping }
  }, [speed, looping])

  const position = useCallback((rigTime: number, poseIndex?: number) => {
    const pose = poseIndex === undefined ? undefined : checkpoints[poseIndex]
    session.time = pose?.rigTime ?? Math.min(duration, Math.max(0, rigTime))
    session.next = 0
    if (pose && poseIndex !== undefined) {
      session.next = poseIndex + 1
    } else if (session.time > 0) {
      const next = checkpoints.findIndex((item) => item.rigTime > session.time + TIME_EPSILON)
      session.next = next === -1 ? checkpoints.length : next
    }
    session.videoAtPose = false
    session.holdElapsed = 0
    session.waitingForStart = true
    session.finished = false
    setTime(session.time)
    setStatus(pose ? `Key pose ${session.next}/${checkpoints.length}` : '')

    const video = videoRef.current
    if (video && videoDuration > 0) {
      const videoTime = pose?.videoTime ?? (
        comparing && session.time > 0
          ? teacherTimeAt(checkpoints, session.time, duration, videoDuration)
          : session.time
      )
      if (Math.abs(video.currentTime - videoTime) > TIME_EPSILON) video.currentTime = videoTime
    }
  }, [checkpoints, comparing, duration, session, videoDuration, videoRef])

  const pause = useCallback(() => {
    session.running = false
    session.waitingForStart = true
    videoRef.current?.pause()
    setPlaying(false)
  }, [session, videoRef])

  useLayoutEffect(() => {
    pause()
    if (duration > 0) position(session.time)
    else setStatus('')
  }, [duration, enabled, pause, position, session, source])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !playing || !enabled) return

    let raf = 0
    let previousFrame: number | undefined
    let heldLastFrame = false
    let playPending = false
    let disposed = false

    const startVideo = () => {
      if (playPending || !video.paused || video.seeking || video.ended) return
      playPending = true
      void video.play().catch((error: DOMException) => {
        if (disposed || !session.running || error.name === 'AbortError') return
        pause()
        onError('Playback could not start.')
      }).finally(() => {
        playPending = false
      })
    }

    const tick = (now: number) => {
      if (!session.running) return
      raf = requestAnimationFrame(tick)
      const elapsed = previousFrame === undefined ? 0 : now - previousFrame
      previousFrame = now

      const ready = !video.seeking && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
      if (video.error) return
      const { speed: rate, looping: loop } = options.current
      if (video.playbackRate !== rate) video.playbackRate = rate
      if (session.waitingForStart && !ready) {
        if (!session.videoAtPose) startVideo()
        return
      }

      const pose = checkpoints[session.next]
      if (pose && !session.videoAtPose && ready && video.currentTime >= pose.videoTime - TIME_EPSILON) {
        video.pause()
        session.videoAtPose = true
        if (Math.abs(video.currentTime - pose.videoTime) > TIME_EPSILON) {
          video.currentTime = pose.videoTime
        }
      }
      if (!session.videoAtPose) startVideo()

      const advance = session.waitingForStart ? 0 : Math.min(elapsed, 100) * rate / 1000
      session.waitingForStart = false
      const rigTarget = pose?.rigTime ?? duration
      session.time = comparing
        ? Math.min(rigTarget, session.time + advance)
        : session.videoAtPose || video.ended ? rigTarget : Math.min(rigTarget, video.currentTime)
      setTime(session.time)

      if (!pose) {
        setStatus('')
        if (session.time >= duration && (video.ended || video.currentTime >= videoDuration)) {
          if (loop) {
            position(0)
          } else {
            session.finished = true
            pause()
          }
        }
        return
      }

      const rigAtPose = session.time >= rigTarget
      const videoAtPose = session.videoAtPose && !video.seeking &&
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
      const label = `Key pose ${session.next + 1}/${checkpoints.length}`
      if (!rigAtPose || !videoAtPose) {
        heldLastFrame = false
        const waiting = comparing && (videoAtPose || rigAtPose)
          ? videoAtPose ? 'Video waiting for 3D rig' : '3D rig waiting for video'
          : 'Moving to pose'
        setStatus(`${label} · ${waiting}`)
        return
      }

      setStatus(`${label} · Holding both`)
      if (heldLastFrame) session.holdElapsed += elapsed
      heldLastFrame = true
      if (session.holdElapsed >= HOLD_MS) {
        session.next += 1
        session.videoAtPose = false
        session.holdElapsed = 0
        heldLastFrame = false
      }
    }

    raf = requestAnimationFrame(tick)
    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      video.pause()
    }
  }, [checkpoints, comparing, duration, enabled, onError, pause, playing, position, session, videoDuration, videoRef])

  const toggle = () => {
    if (!enabled) return
    if (session.running) {
      pause()
    } else {
      if (session.finished) position(0)
      session.running = true
      setPlaying(true)
    }
  }

  const seek = (nextTime: number, snap = false) => {
    pause()
    const nearest = snap && checkpoints.length > 0
      ? checkpoints.reduce((best, pose, index) =>
          Math.abs(pose.rigTime - nextTime) < Math.abs(checkpoints[best].rigTime - nextTime) ? index : best, 0)
      : undefined
    position(nextTime, nearest)
  }

  const step = (delta: -1 | 1) => {
    const upcoming = checkpoints[session.next]
    const atUpcoming = session.videoAtPose && upcoming && session.time >= upcoming.rigTime
    const previous = checkpoints[session.next - 1]
    const atPrevious = previous && Math.abs(session.time - previous.rigTime) <= TIME_EPSILON
    const index = delta === 1
      ? session.next + (atUpcoming ? 1 : 0)
      : session.next - (atPrevious && !atUpcoming ? 2 : 1)
    const target = Math.max(0, Math.min(checkpoints.length - 1, index))
    pause()
    position(0, target)
  }

  return { time, playing, status, pause, toggle, seek, step }
}
