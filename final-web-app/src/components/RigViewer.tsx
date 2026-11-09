import { useEffect, useRef, useState, type RefObject } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import type { Feedback } from '../data/api'

const boneByJoint: Record<string, string> = {
  L_armpit: 'LeftArm', R_armpit: 'RightArm', L_elbow: 'LeftForeArm', R_elbow: 'RightForeArm',
  L_pelvis: 'LeftUpLeg', R_pelvis: 'RightUpLeg', L_knee: 'LeftLeg', R_knee: 'RightLeg',
  chest_tilt: 'Spine2', hip_tilt: 'Hips',
}

function activeFeedback(entries: Feedback[], time: number) {
  let low = 0
  let high = entries.length
  while (low < high) {
    const mid = (low + high) >>> 1
    if (entries[mid].end <= time) low = mid + 1
    else high = mid
  }
  return entries[low]?.start <= time ? entries[low].joints : []
}

export default function RigViewer({ url, video, feedback }: {
  url: string
  video: RefObject<HTMLVideoElement>
  feedback: Feedback[]
}) {
  const host = useRef<HTMLDivElement>(null)
  const feedbackRef = useRef(feedback)
  const [status, setStatus] = useState('Loading rig…')
  useEffect(() => { feedbackRef.current = feedback }, [feedback])

  useEffect(() => {
    const el = host.current!
    const w = () => el.clientWidth || 600
    const h = () => el.clientHeight || 600

    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x0b0a08, 0.05)
    const camera = new THREE.PerspectiveCamera(45, w() / h(), 0.1, 200)
    camera.position.set(0, 1.4, 4)
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }) }
    catch { setStatus('WebGL is unavailable in this browser.'); return }
    setStatus('Loading rig…')
    renderer.setSize(w(), h())
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.shadowMap.enabled = true
    el.appendChild(renderer.domElement)

    scene.add(new THREE.HemisphereLight(0xffe0b0, 0x1a1208, 1.1))
    const key = new THREE.DirectionalLight(0xffd58a, 2.2)
    key.position.set(3, 5, 4)
    key.castShadow = true
    scene.add(key)

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(4, 64),
      new THREE.MeshStandardMaterial({ color: 0x1a1710, roughness: 1 }),
    )
    floor.rotation.x = -Math.PI / 2
    floor.receiveShadow = true
    scene.add(floor)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 1, 0)
    controls.enableDamping = true
    controls.update()

    const bones = new Map<string, THREE.Bone>()
    const markerGeometry = new THREE.SphereGeometry(0.055, 16, 12)
    const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff2828, depthTest: false, toneMapped: false })
    const markers = Array.from({ length: 2 }, () => {
      const marker = new THREE.Mesh(markerGeometry, markerMaterial)
      marker.visible = false
      marker.renderOrder = 1
      scene.add(marker)
      return marker
    })
    let mixer: THREE.AnimationMixer | null = null
    let animation: THREE.AnimationAction | null = null
    let model: THREE.Object3D | undefined
    let disposed = false
    const disposeModel = (root: THREE.Object3D) => root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return
      object.geometry.dispose()
      for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
        Object.values(material).forEach((value) => { if (value instanceof THREE.Texture) value.dispose() })
        material.dispose()
      }
      if (object instanceof THREE.SkinnedMesh) object.skeleton.dispose()
    })
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (disposed) { disposeModel(gltf.scene); return }
        model = gltf.scene
        const box = new THREE.Box3().setFromObject(model)
        const size = box.getSize(new THREE.Vector3())
        const s = 1.8 / Math.max(size.y, 0.001)
        model.scale.setScalar(s)
        const b2 = new THREE.Box3().setFromObject(model)
        model.position.y -= b2.min.y
        model.position.x -= (b2.min.x + b2.max.x) / 2
        model.position.z -= (b2.min.z + b2.max.z) / 2
        model.traverse((object) => {
          if (object instanceof THREE.Mesh) object.castShadow = true
          if (object instanceof THREE.Bone) {
            const name = object.name.slice(object.name.lastIndexOf(':') + 1)
            bones.set(name.replace(/^mixamorig\d*[_:]?/i, ''), object)
          }
        })
        scene.add(model)
        if (gltf.animations.length) {
          mixer = new THREE.AnimationMixer(model)
          animation = mixer.clipAction(gltf.animations[0])
          animation.setLoop(THREE.LoopOnce, 1).play()
          animation.clampWhenFinished = true
          animation.paused = true
        }
        setStatus('')
      },
      (e) => !disposed && e.total && setStatus(`Loading rig… ${Math.round((e.loaded / e.total) * 100)}%`),
      () => { if (!disposed) setStatus('Could not load the generated rig.') },
    )

    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const time = video.current?.currentTime ?? 0
      if (animation && mixer) {
        animation.time = Math.min(time, animation.getClip().duration)
        mixer.update(0)
      }
      const joints = activeFeedback(feedbackRef.current, time)
      markers.forEach((marker, index) => {
        const bone = joints[index] && bones.get(boneByJoint[joints[index].id])
        marker.visible = !!bone
        bone?.getWorldPosition(marker.position)
      })
      controls.update()
      renderer.render(scene, camera)
    }
    loop()

    const ro = new ResizeObserver(() => {
      renderer.setSize(w(), h())
      camera.aspect = w() / h()
      camera.updateProjectionMatrix()
    })
    ro.observe(el)

    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      mixer?.stopAllAction()
      if (model) disposeModel(model)
      floor.geometry.dispose()
      floor.material.dispose()
      markerGeometry.dispose()
      markerMaterial.dispose()
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [url, video])

  return (
    <div ref={host} className="relative h-full w-full">
      {status && (
        <div className="stencil absolute inset-0 grid place-items-center text-sm text-cream/60">{status}</div>
      )}
    </div>
  )
}
