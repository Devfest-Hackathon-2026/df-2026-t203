import type { ReactNode } from 'react'

/** Fixed main-background image at 50% opacity over near-black. */
export default function PageBackground({ children }: { children: ReactNode }) {
  return (
    <div className="relative flex h-full overflow-hidden bg-ink">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-50"
        style={{ backgroundImage: "url('/media/main-background.png')" }}
      />
      <div className="relative z-10 flex h-full w-full">{children}</div>
    </div>
  )
}
