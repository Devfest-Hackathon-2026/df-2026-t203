import { useEffect, useRef, useState } from 'react'
import type { JointError, Recording } from './api'

type Phase = 'IDLE' | 'STARTING' | 'SEEKING' | 'PAUSED' | 'PLAYING' | 'SAVING'

interface Feedback {
  state: Phase
  pose: number
  total: number
  hold: number
  required_hold: number
  score: number | null
  dtw: number | null
  joints: JointError[]
  arrows: { start: [number, number]; end: [number, number] }[]
}

interface Message extends Partial<Feedback> {
  type?: string
  action?: string
  seek_time?: number
  target_time?: number
  message?: string
  recording?: Recording
}

interface Session {
  phase: Phase
  stream?: MediaStream
  socket?: WebSocket
  captureTimer?: number
  playTimer?: number
  cancelSeek?: () => void
  inFlight: boolean
  generation: number
}

export function useTraining(onSaved: (recording: Recording) => void) {
  const teacher = useRef<HTMLVideoElement>(null)
  const camera = useRef<HTMLVideoElement>(null)
  const session = useRef<Session>()
  const [phase, setPhase] = useState<Phase>('IDLE')
  const [feedback, setFeedback] = useState<Feedback>()
  const [dimensions, setDimensions] = useState<[number, number]>([1280, 720])
  const [error, setError] = useState('')

  function dispose() {
    const take = session.current
    session.current = undefined
    if (!take) return
    window.clearInterval(take.captureTimer)
    if (take.playTimer !== undefined) cancelAnimationFrame(take.playTimer)
    take.cancelSeek?.()
    take.socket?.close()
    take.stream?.getTracks().forEach((track) => track.stop())
    if (camera.current) camera.current.srcObject = null
    teacher.current?.pause()
  }

  useEffect(() => dispose, [])

  function finish(message = '') {
    dispose()
    setPhase('IDLE')
    setFeedback(undefined)
    setError(message)
  }

  function setState(take: Session, next: Phase) {
    if (next === 'SEEKING') take.generation += 1
    take.phase = next
    setPhase(next)
    if (next === 'SEEKING' || next === 'SAVING') setFeedback(undefined)
  }

  function send(take: Session, type: string) {
    if (take.socket?.readyState === WebSocket.OPEN) take.socket.send(JSON.stringify({ type }))
  }

  function cancelPlayback(take: Session) {
    if (take.playTimer !== undefined) cancelAnimationFrame(take.playTimer)
    take.playTimer = undefined
    teacher.current?.pause()
  }

  async function handleMessage(take: Session, message: Message) {
    const video = teacher.current
    if (session.current !== take || !video) return
    if (message.type === 'SESSION_COMPLETE' && message.recording) {
      finish()
      onSaved(message.recording)
      return
    }
    if (message.type === 'ERROR' || message.type === 'SESSION_CANCELLED') {
      finish(message.message || 'Training ended before a recording was saved.')
      return
    }
    if (message.type === 'SAVING') {
      cancelPlayback(take)
      window.clearInterval(take.captureTimer)
      take.stream?.getTracks().forEach((track) => track.stop())
      if (camera.current) camera.current.srcObject = null
      setState(take, 'SAVING')
      return
    }
    if (message.type === 'TRAINING_FEEDBACK') {
      take.inFlight = false
      if (take.phase === 'SAVING' || take.phase === 'SEEKING') return
      if (message.state) setState(take, message.state)
      setFeedback(message as Feedback)
      return
    }
    if (take.phase === 'SAVING') return
    if (message.action === 'seek' && message.seek_time !== undefined) {
      cancelPlayback(take)
      take.cancelSeek?.()
      setState(take, 'SEEKING')
      const target = message.seek_time
      const ready = () => {
        take.cancelSeek?.()
        if (session.current === take && take.phase === 'SEEKING') send(take, 'SEEKED')
      }
      if (Math.abs(video.currentTime - target) < 0.002 && !video.seeking) ready()
      else {
        video.addEventListener('seeked', ready, { once: true })
        take.cancelSeek = () => { video.removeEventListener('seeked', ready); take.cancelSeek = undefined }
        video.currentTime = target
      }
    } else if (message.action === 'pause') {
      cancelPlayback(take)
      if (message.state) setState(take, message.state)
    } else if (message.action === 'play_until' && message.target_time !== undefined) {
      setState(take, 'PLAYING')
      const target = Math.min(message.target_time, video.duration)
      const generation = take.generation
      await video.play()
      const watch = () => {
        if (session.current !== take || take.generation !== generation || take.phase !== 'PLAYING') return
        if (video.currentTime >= target || video.ended) {
          cancelPlayback(take)
          video.currentTime = target
          setState(take, 'PAUSED')
          send(take, 'ARRIVED')
        } else take.playTimer = requestAnimationFrame(watch)
      }
      watch()
    }
  }

  async function start(videoId: string) {
    if (session.current) return
    setError('')
    if (!teacher.current || teacher.current.readyState < 1) {
      setError('Wait for the teacher video to load before starting.')
      return
    }
    const take: Session = { phase: 'STARTING', inFlight: false, generation: 0 }
    session.current = take
    setState(take, 'STARTING')
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost.')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 } } })
      if (session.current !== take) { stream.getTracks().forEach((track) => track.stop()); return }
      take.stream = stream
      const preview = camera.current
      if (!preview) throw new Error('Camera preview is unavailable.')
      preview.srcObject = stream
      await preview.play()
      if (session.current !== take) return
      stream.getVideoTracks()[0].addEventListener('ended', () => {
        if (session.current === take && take.phase !== 'SAVING') finish('The camera disconnected. Start a new take to try again.')
      })

      const canvas = document.createElement('canvas')
      const scale = Math.min(1, 1280 / Math.max(preview.videoWidth, preview.videoHeight))
      canvas.width = 2 * Math.floor(preview.videoWidth * scale / 2)
      canvas.height = 2 * Math.floor(preview.videoHeight * scale / 2)
      setDimensions([canvas.width, canvas.height])
      const context = canvas.getContext('2d', { alpha: false })
      if (!context || !canvas.width || !canvas.height) throw new Error('Could not capture camera frames.')
      const origin = location.origin.replace(/^http/, 'ws')
      const ws = new WebSocket(`${origin}/ws/training/${encodeURIComponent(videoId)}?camera=browser`)
      take.socket = ws
      ws.onmessage = (event) => {
        void handleMessage(take, JSON.parse(event.data) as Message).catch(() => {
          if (session.current === take) finish('The teacher video could not continue. Check that playback is allowed and retry.')
        })
      }
      ws.onclose = () => {
        if (session.current === take) finish('Training disconnected. Check the library for saved takes before trying again.')
      }
      ws.onopen = () => {
        if (session.current !== take) return
        const epoch = performance.now()
        take.captureTimer = window.setInterval(() => {
          if (session.current !== take || take.inFlight || ws.bufferedAmount || !['PAUSED', 'PLAYING'].includes(take.phase)) return
          take.inFlight = true
          const generation = take.generation
          const stamp = new ArrayBuffer(4)
          new DataView(stamp).setUint32(0, Math.round(performance.now() - epoch))
          context.drawImage(preview, 0, 0, canvas.width, canvas.height)
          canvas.toBlob((jpeg) => {
            if (session.current !== take || generation !== take.generation || ws.readyState !== WebSocket.OPEN || !['PAUSED', 'PLAYING'].includes(take.phase)) {
              take.inFlight = false
              return
            }
            if (!jpeg || jpeg.size + 4 > 512 * 1024) { finish('The camera frame could not be encoded.'); return }
            ws.send(new Blob([stamp, jpeg]))
          }, 'image/jpeg', 0.8)
        }, 1000 / 24)
      }
    } catch (err) {
      if (session.current !== take) return
      finish(err instanceof DOMException && err.name === 'NotAllowedError'
        ? 'Camera permission was denied. Allow access in your browser to start training.' : (err as Error).message)
    }
  }

  function control(type: string) {
    const take = session.current
    if (!take) return
    if (type === 'STOP_SESSION') {
      if (take.socket?.readyState !== WebSocket.OPEN) { finish(); return }
      setState(take, 'SAVING')
      cancelPlayback(take)
    } else if (take.phase === 'SEEKING' || take.phase === 'SAVING') return
    send(take, type)
  }

  return { teacher, camera, phase, feedback, dimensions, error, start, control }
}
