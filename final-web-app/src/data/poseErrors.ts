import type { Feedback } from './api'

const MAJOR_ERROR_DEGREES = 25
const MIN_ERROR_SECONDS = 2
const TIMESTAMP_EPSILON = 0.000001

const jointsByRegion = {
  'Left arm': ['L_armpit', 'L_elbow'],
  'Right arm': ['R_armpit', 'R_elbow'],
  'Left leg': ['L_pelvis', 'L_knee'],
  'Right leg': ['R_pelvis', 'R_knee'],
  Torso: ['chest_tilt', 'hip_tilt'],
}

export type PoseErrorRegion = keyof typeof jointsByRegion

export const mirroredPoseErrorRegion: Record<PoseErrorRegion, PoseErrorRegion> = {
  'Left arm': 'Right arm',
  'Right arm': 'Left arm',
  'Left leg': 'Right leg',
  'Right leg': 'Left leg',
  Torso: 'Torso',
}

interface ErrorRun {
  start: number
  end: number
  severity: number
}

export interface RegionErrors {
  region: PoseErrorRegion
  runs: ErrorRun[]
}

export function sustainedPoseErrors(feedback: Feedback[]): RegionErrors[] {
  const entries = feedback
    .filter(({ start, end }) => Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end > start)
    .sort((a, b) => a.start - b.start)
  const regions = Object.keys(jointsByRegion) as PoseErrorRegion[]

  return regions.map((region) => {
    const runs: ErrorRun[] = []
    let pending: { start: number; end: number; totalDegrees: number } | undefined
    const finishRun = () => {
      if (!pending) return
      const { start, end, totalDegrees } = pending
      if (end - start >= MIN_ERROR_SECONDS - TIMESTAMP_EPSILON) {
        runs.push({ start, end, severity: totalDegrees / (end - start) })
      }
      pending = undefined
    }

    for (const [index, entry] of entries.entries()) {
      const start = entry.start
      const end = Math.min(entry.end, entries[index + 1]?.start ?? entry.end)
      if (end <= start) continue

      const degrees = entry.joints.reduce((worst, joint) => (
        jointsByRegion[region].includes(joint.id) && Number.isFinite(joint.degrees)
          ? Math.max(worst, joint.degrees)
          : worst
      ), 0)
      if (degrees < MAJOR_ERROR_DEGREES) {
        finishRun()
        continue
      }

      if (pending && start - pending.end > TIMESTAMP_EPSILON) finishRun()
      pending ??= { start, end, totalDegrees: 0 }
      pending.end = end
      pending.totalDegrees += degrees * (end - start)
    }
    finishRun()
    return { region, runs }
  }).filter(({ runs }) => runs.length > 0)
}

export function poseErrorsAt(timeline: RegionErrors[], time: number): PoseErrorRegion[] {
  const active = []
  for (const { region, runs } of timeline) {
    let low = 0
    let high = runs.length
    while (low < high) {
      const mid = (low + high) >>> 1
      if (runs[mid].end <= time) low = mid + 1
      else high = mid
    }
    const run = runs[low]
    if (run && run.start <= time) active.push({ region, severity: run.severity })
  }
  return active.sort((a, b) => b.severity - a.severity).slice(0, 3).map(({ region }) => region)
}
