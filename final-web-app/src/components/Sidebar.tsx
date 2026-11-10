import { useRef, useState } from 'react'
import type { Recording, Teacher } from '../data/api'

interface Props {
  teachers: Teacher[]
  recordings: Recording[]
  selected: string | null
  onTeacher?: (teacher: Teacher) => void
  onRecording: (recording: Recording) => void
  onDeleteRecording: (recording: Recording) => Promise<void>
  onUpload: (file: File) => Promise<void>
  onYouTube?: (url: string) => Promise<void>
  disabled?: boolean
  loading: boolean
}

export default function Sidebar({
  teachers,
  recordings,
  selected,
  onTeacher,
  onRecording,
  onDeleteRecording,
  onUpload,
  onYouTube,
  disabled,
  loading,
}: Props) {
  const [collapsed, setCollapsed] = useState(() => window.matchMedia('(max-width: 639px)').matches)
  const [addOpen, setAddOpen] = useState(false)
  const [youtube, setYoutube] = useState(false)
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState('')
  const [closedFolders, setClosedFolders] = useState<Record<string, boolean>>({})
  const fileInputRef = useRef<HTMLInputElement>(null)

  const toggleFolder = (folderId: string) => {
    setClosedFolders((prev) => ({ ...prev, [folderId]: !prev[folderId] }))
  }

  const submit = async (action: () => Promise<void>) => {
    setBusy(true)
    setError('')
    try {
      await action()
      setYoutube(false)
      setAddOpen(false)
      setUrl('')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const confirmDelete = async (recording: Recording) => {
    setDeleting(true)
    setError('')
    try {
      await onDeleteRecording(recording)
      setDeleteTarget(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setDeleting(false)
    }
  }

  const renderRecording = (recording: Recording) => (
    <li key={recording.id}>
      <div className="flex items-start gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onRecording(recording)}
          aria-current={selected === recording.id ? 'true' : undefined}
          className="min-w-0 flex-1 py-1 text-left text-sm text-cream/80 transition hover:text-acid aria-[current=true]:font-medium aria-[current=true]:text-amber disabled:opacity-50"
        >
          {recording.title}
        </button>
        <button
          type="button"
          disabled={disabled || deleting || recording.status === 'processing'}
          aria-label={`Delete ${recording.title}`}
          aria-expanded={deleteTarget === recording.id}
          title={recording.status === 'processing' ? 'Recording is processing' : 'Delete recording'}
          onClick={() => {
            setError('')
            setDeleteTarget(deleteTarget === recording.id ? null : recording.id)
          }}
          className="shrink-0 px-1 py-1.5 text-xs text-red-300 hover:text-red-100 disabled:opacity-50"
        >
          Delete
        </button>
      </div>
      {deleteTarget === recording.id && (
        <div role="group" aria-label={`Confirm deletion of ${recording.title}`} className="mt-1 space-y-2 border border-edge bg-ink/80 p-2 text-xs">
          <p className="text-cream/80">Permanently delete this recording and its generated files?</p>
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={deleting} onClick={() => setDeleteTarget(null)} className="text-cream hover:text-acid disabled:opacity-50">
              Cancel
            </button>
            <button
              type="button"
              disabled={disabled || deleting || recording.status === 'processing'}
              onClick={() => void confirmDelete(recording)}
              className="font-medium text-red-300 hover:text-red-100 disabled:opacity-50"
            >
              {deleting ? 'Deleting…' : 'Delete permanently'}
            </button>
          </div>
        </div>
      )}
    </li>
  )

  const standalone = recordings.filter(
    (recording) => !teachers.some((teacher) => teacher.id === recording.video_id),
  )

  return (
    <aside
      className={`panel relative z-20 flex shrink-0 flex-col border-y-0 border-l-0 ${
        collapsed ? 'w-14' : 'w-56 sm:w-64'
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,.mp4,.mov,.webm"
        disabled={busy || disabled}
        className="hidden"
        onChange={(event) => {
          const file = event.currentTarget.files?.[0]
          event.currentTarget.value = ''
          if (file) void submit(() => onUpload(file))
        }}
      />

      <div className={`flex items-center ${collapsed ? 'justify-center' : 'justify-between'} p-3`}>
        {!collapsed && (
          <button
            type="button"
            disabled={busy || disabled}
            onClick={() => {
              if (onYouTube) {
                setAddOpen((prev) => !prev)
              } else {
                fileInputRef.current?.click()
              }
            }}
            aria-label={onYouTube ? 'Add teacher video' : 'Upload recording'}
            title={onYouTube ? 'Add teacher video' : 'Upload recording'}
            className="grid h-9 w-9 place-items-center border border-edge bg-ink/60 text-lg text-cream transition hover:border-acid hover:text-acid disabled:opacity-50"
          >
            +
          </button>
        )}
        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          aria-label="Toggle library"
          aria-expanded={!collapsed}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="grid h-9 w-9 place-items-center border border-edge bg-ink/60 text-cream transition hover:border-acid hover:text-acid"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
          >
            <rect x="3" y="4" width="18" height="16" rx="1" />
            <line x1="9" y1="4" x2="9" y2="20" />
            <path d={collapsed ? 'M13 10l3 2-3 2' : 'M15 10l-3 2 3 2'} />
          </svg>
        </button>
      </div>

      {!collapsed && (
        <>
          {(addOpen || busy || error) && (
            <div className="space-y-2.5 border-y border-edge bg-ink/80 px-3 py-3 text-xs">
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={busy || disabled}
                  onClick={() => fileInputRef.current?.click()}
                  className="ctrl-btn"
                >
                  {onYouTube ? 'Upload Video' : 'Upload Recording'}
                </button>
                {onYouTube && (
                  <button
                    type="button"
                    disabled={busy || disabled}
                    onClick={() => setYoutube((prev) => !prev)}
                    className={`ctrl-btn ${youtube ? 'active' : ''}`}
                  >
                    YouTube Link
                  </button>
                )}
              </div>
              {youtube && onYouTube && (
                <form
                  className="space-y-2 pt-1"
                  onSubmit={(event) => {
                    event.preventDefault()
                    if (url.trim()) void submit(() => onYouTube(url.trim()))
                  }}
                >
                  <input
                    type="url"
                    required
                    aria-label="YouTube URL"
                    value={url}
                    onChange={(event) => setUrl(event.target.value)}
                    disabled={busy || disabled}
                    className="w-full border border-edge bg-ink px-2.5 py-1.5 text-xs text-cream"
                    placeholder="https://youtu.be/…"
                  />
                  <button type="submit" className="ctrl-btn" disabled={busy || disabled}>
                    Import
                  </button>
                </form>
              )}
              {busy && (
                <p role="status" className="text-amber">
                  Importing video…
                </p>
              )}
              {error && (
                <p role="alert" className="text-red-300">
                  {error}
                </p>
              )}
            </div>
          )}

          <nav aria-label="Media library" className="flex-1 space-y-4 overflow-y-auto px-4 py-3">
            {loading && <p className="py-2 text-sm text-cream/60">Loading library…</p>}
            {!loading && !teachers.length && !recordings.length && (
              <p className="py-2 text-sm text-cream/60">Click + to upload a video.</p>
            )}

            {teachers.map((teacher) => {
              const takes = recordings.filter((recording) => recording.video_id === teacher.id)
              if (!onTeacher && !takes.length) return null
              const isOpen = !closedFolders[teacher.id]
              const isSelectedTeacher = selected === teacher.id

              return (
                <section key={teacher.id}>
                  <div className="flex items-center justify-between gap-2">
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() => {
                        if (onTeacher) {
                          onTeacher(teacher)
                        } else {
                          toggleFolder(teacher.id)
                        }
                      }}
                      aria-current={isSelectedTeacher ? 'true' : undefined}
                      className="flex min-w-0 flex-1 items-center gap-2.5 text-left font-display text-xl text-cream transition hover:text-acid aria-[current=true]:text-amber disabled:opacity-50"
                    >
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.7"
                        className="shrink-0 text-cream/85"
                      >
                        <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
                      </svg>
                      <span className="truncate">{teacher.name}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => toggleFolder(teacher.id)}
                      aria-label={isOpen ? `Collapse ${teacher.name}` : `Expand ${teacher.name}`}
                      className="px-1 text-xs text-cream/70 hover:text-acid"
                    >
                      {isOpen ? '∧' : '∨'}
                    </button>
                  </div>

                  {isOpen && (
                    <ul className="ml-2 mt-2 space-y-1.5 border-l border-edge pl-4">
                      {onTeacher && (
                        <li>
                          <button
                            type="button"
                            disabled={disabled}
                            onClick={() => onTeacher(teacher)}
                            aria-current={isSelectedTeacher ? 'true' : undefined}
                            className="w-full py-1 text-left text-sm text-cream/80 transition hover:text-acid aria-[current=true]:font-medium aria-[current=true]:text-amber disabled:opacity-50"
                          >
                            {teacher.name}
                          </button>
                        </li>
                      )}
                      {takes.map(renderRecording)}
                    </ul>
                  )}
                </section>
              )
            })}

            {standalone.length > 0 && (
              <section>
                <div className="flex items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => toggleFolder('standalone')}
                    className="flex min-w-0 flex-1 items-center gap-2.5 text-left font-display text-xl text-cream transition hover:text-acid"
                  >
                    <svg
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      className="shrink-0 text-cream/85"
                    >
                      <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
                    </svg>
                    <span className="truncate">uploads</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => toggleFolder('standalone')}
                    aria-label="Toggle uploads folder"
                    className="px-1 text-xs text-cream/70 hover:text-acid"
                  >
                    {!closedFolders['standalone'] ? '∧' : '∨'}
                  </button>
                </div>
                {!closedFolders['standalone'] && (
                  <ul className="ml-2 mt-2 space-y-1.5 border-l border-edge pl-4">
                    {standalone.map(renderRecording)}
                  </ul>
                )}
              </section>
            )}
          </nav>
        </>
      )}
    </aside>
  )
}
