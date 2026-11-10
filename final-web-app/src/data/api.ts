import { useCallback, useEffect, useRef, useState } from 'react'

export interface Teacher {
  id: string
  name: string
  filename: string
  status: string
  is_processed: boolean
  keypose_count: number
  error_message?: string
}

export interface KeyposeSync {
  teacher_time: number
  mocap_arrive_time: number
  mocap_match_time: number
  next_teacher_time: number
  next_mocap_arrive_time: number
}

export interface Recording {
  id: string
  video_id: string
  video_name: string
  title: string
  filename: string
  status: string
  is_processed?: boolean
  output_dir_name: string
  glb_filename?: string
  mocap_mirrored?: boolean
  blend_filename?: string
  fbx_filename?: string
  teacher_video_filename?: string
  error_message?: string
  keypose_sync?: KeyposeSync[]
}

export interface JointError { id: string; degrees: number }
export interface Feedback { start: number; end: number; joints: JointError[] }

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' ? body.detail : `Request failed (${response.status}).`)
  }
  return response.json()
}

export function post<T>(path: string, body?: object) {
  return request<T>(path, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
}

export function upload<T>(path: string, file: File) {
  const body = new FormData()
  body.append('video', file)
  return request<T>(`${path}?auto_process=true`, { method: 'POST', body })
}

export function useLibrary() {
  const [teachers, setTeachers] = useState<Teacher[]>([])
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const recordingsRevision = useRef(0)
  const refresh = useCallback(async (signal?: AbortSignal) => {
    const revision = recordingsRevision.current
    try {
      const [videos, takes] = await Promise.all([
        request<Teacher[]>('/api/videos', { signal }),
        request<Recording[]>('/api/recordings', { signal }),
      ])
      if (signal?.aborted) return
      setTeachers(videos)
      if (revision === recordingsRevision.current) setRecordings(takes)
      setError('')
    } catch (err) {
      if (!signal?.aborted) setError((err as Error).message)
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [])

  const deleteRecording = useCallback(async (recording: Recording) => {
    const result = await request<{ recordings: Recording[] }>(
      `/api/recordings/${encodeURIComponent(recording.id)}`,
      { method: 'DELETE' },
    )
    // A poll started before the delete may still contain the removed recording.
    recordingsRevision.current += 1
    setRecordings(result.recordings)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    let timer: number
    const poll = async () => {
      await refresh(controller.signal)
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 2000)
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [refresh])

  return { teachers, recordings, error, loading, refresh, deleteRecording }
}

export const teacherUrl = (filename: string) => `/teacher_videos/${encodeURIComponent(filename)}`
export const recordingUrl = (filename: string) => `/recordings/${encodeURIComponent(filename)}`
export const outputUrl = (recording: Recording, filename: string) =>
  `/outputs/${encodeURIComponent(recording.output_dir_name)}/${encodeURIComponent(filename)}`
