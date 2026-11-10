import * as THREE from 'three'
import type { PoseErrorRegion } from '../data/poseErrors'

const regionBones: Record<PoseErrorRegion, string[]> = {
  'Left arm': ['LeftArm', 'LeftForeArm', 'LeftHand'],
  'Right arm': ['RightArm', 'RightForeArm', 'RightHand'],
  'Left leg': ['LeftUpLeg', 'LeftLeg', 'LeftFoot'],
  'Right leg': ['RightUpLeg', 'RightLeg', 'RightFoot'],
  Torso: ['Hips', 'Spine2'],
}

export function createPoseErrorHighlights(scene: THREE.Scene, model: THREE.Object3D) {
  const bones = new Map<string, THREE.Bone>()
  model.traverse((object) => {
    if (!(object instanceof THREE.Bone)) return
    const name = object.name.slice(object.name.lastIndexOf(':') + 1)
    bones.set(name.replace(/^mixamorig\d*[_:]?/i, ''), object)
  })

  const geometry = new THREE.CapsuleGeometry(0.5, 1, 6, 12)
  const coreMaterial = new THREE.MeshBasicMaterial({
    color: 0xff2828, transparent: true, opacity: 0.36, depthWrite: false, toneMapped: false,
  })
  const haloMaterial = new THREE.MeshBasicMaterial({
    color: 0xff2828, transparent: true, opacity: 0.12, depthWrite: false, toneMapped: false,
  })
  const highlights: {
    region: PoseErrorRegion
    from: THREE.Bone
    to: THREE.Bone
    group: THREE.Group
  }[] = []

  for (const region of Object.keys(regionBones) as PoseErrorRegion[]) {
    const names = regionBones[region]
    for (let index = 1; index < names.length; index += 1) {
      const from = bones.get(names[index - 1])
      const to = bones.get(names[index])
      if (!from || !to) continue
      const group = new THREE.Group()
      group.visible = false
      const halo = new THREE.Mesh(geometry, haloMaterial)
      halo.scale.set(1.22, 1.05, 1.22)
      group.add(halo, new THREE.Mesh(geometry, coreMaterial))
      scene.add(group)
      highlights.push({ region, from, to, group })
    }
  }

  const start = new THREE.Vector3()
  const end = new THREE.Vector3()
  const direction = new THREE.Vector3()
  const up = new THREE.Vector3(0, 1, 0)

  return {
    update(regions: PoseErrorRegion[]) {
      for (const { region, from, to, group } of highlights) {
        group.visible = regions.includes(region)
        if (!group.visible) continue
        from.getWorldPosition(start)
        to.getWorldPosition(end)
        direction.subVectors(end, start)
        const length = direction.length()
        if (length <= 0.0001) {
          group.visible = false
          continue
        }
        group.position.addVectors(start, end).multiplyScalar(0.5)
        group.quaternion.setFromUnitVectors(up, direction.divideScalar(length))
        const diameter = length * (region === 'Torso' ? 1.15 : 0.5)
        group.scale.set(diameter, (length + diameter) * 0.5, diameter)
      }
    },
    dispose() {
      for (const { group } of highlights) scene.remove(group)
      geometry.dispose()
      coreMaterial.dispose()
      haloMaterial.dispose()
    },
  }
}
