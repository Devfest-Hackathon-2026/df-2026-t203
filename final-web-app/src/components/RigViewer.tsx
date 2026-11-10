import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

type CameraPreset = 'front' | 'perspective' | 'side' | 'upper'

export default function RigViewer({
  url,
  video,
  mocapTime,
  onDurationLoaded,
}: {
  url: string
  video?: RefObject<HTMLVideoElement>
  mocapTime?: number
  onDurationLoaded?: (duration: number) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const mocapTimeRef = useRef(mocapTime)
  const durationCallbackRef = useRef(onDurationLoaded)
  const presetHandlerRef = useRef<((preset: CameraPreset) => void) | null>(null)
  const [status, setStatus] = useState('Loading 3D Dojo & Rig…')
  const [activePreset, setActivePreset] = useState<CameraPreset | null>(null)

  useLayoutEffect(() => { mocapTimeRef.current = mocapTime }, [mocapTime])
  useEffect(() => { durationCallbackRef.current = onDurationLoaded }, [onDurationLoaded])

  useEffect(() => {
    const el = host.current
    if (!el) return

    const measure = () => {
      const target = el.parentElement ?? el
      const rect = target.getBoundingClientRect()
      const width = Math.max(1, Math.round(rect.width || target.clientWidth || 640))
      const height = Math.max(1, Math.round(rect.height || target.clientHeight || 360))
      return { width, height }
    }

    const { width: initW, height: initH } = measure()
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x161922)
    scene.fog = new THREE.FogExp2(0x161922, 0.015)

    const avatarScale = 1.75
    const baseY = 0.9 * avatarScale

    const camera = new THREE.PerspectiveCamera(45, initW / initH, 0.1, 200)
    camera.position.set(0, 1.3 * avatarScale, 3.1 * avatarScale)

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' })
    } catch {
      setStatus('WebGL is unavailable in this browser.')
      durationCallbackRef.current?.(0)
      return
    }

    setStatus('Loading 3D Dojo & Rig…')
    renderer.setSize(initW, initH)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.25
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    el.appendChild(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xffffff, 1.4))
    const hemiLight = new THREE.HemisphereLight(0xfff5e6, 0x3b3548, 1.4)
    hemiLight.position.set(0, 10, 0)
    scene.add(hemiLight)

    const dirLight = new THREE.DirectionalLight(0xfff8e7, 2.2)
    dirLight.position.set(3, 6, 4)
    dirLight.castShadow = true
    dirLight.shadow.mapSize.width = 2048
    dirLight.shadow.mapSize.height = 2048
    scene.add(dirLight)

    const backLight = new THREE.DirectionalLight(0x8ab4f8, 1.0)
    backLight.position.set(-3, 4, -4)
    scene.add(backLight)

    const gridHelper = new THREE.GridHelper(12, 24, 0x4f46e5, 0x2a2f3d)
    gridHelper.position.y = 0
    scene.add(gridHelper)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, baseY, 0)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.maxDistance = 25
    controls.minDistance = 0.8
    controls.update()

    const applyPreset = (preset: CameraPreset) => {
      setActivePreset(preset)
      controls.target.set(0, baseY, 0)
      const target = controls.target
      switch (preset) {
        case 'front':
          camera.position.set(target.x, target.y + 0.1 * avatarScale, target.z + 3.0 * avatarScale)
          break
        case 'side':
          camera.position.set(target.x + 3.0 * avatarScale, target.y + 0.1 * avatarScale, target.z)
          break
        case 'perspective':
          camera.position.set(target.x + 2.0 * avatarScale, target.y + 0.8 * avatarScale, target.z + 2.5 * avatarScale)
          break
        case 'upper':
          controls.target.set(0, (0.9 + 0.4) * avatarScale, 0)
          camera.position.set(0, (0.9 + 0.3) * avatarScale, 1.8 * avatarScale)
          break
      }
      controls.update()
    }
    presetHandlerRef.current = applyPreset

    let mixer: THREE.AnimationMixer | null = null
    let animation: THREE.AnimationAction | null = null
    let model: THREE.Object3D | undefined
    let dojoModel: THREE.Object3D | undefined
    let disposed = false

    const disposeModel = (root: THREE.Object3D) =>
      root.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return
        object.geometry.dispose()
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
          Object.values(material).forEach((value) => {
            if (value instanceof THREE.Texture) value.dispose()
          })
          material.dispose()
        }
        if (object instanceof THREE.SkinnedMesh) object.skeleton.dispose()
      })

    const loader = new GLTFLoader()

    // 1. Load the Dojo 3D environment
    const loadDojo = (dojoUrl: string, allowFallback: boolean) => {
      loader.load(
        dojoUrl,
        (gltf) => {
          if (disposed) {
            disposeModel(gltf.scene)
            return
          }
          const envScene = gltf.scene
          const dojoB = envScene.getObjectByName('Dojo_B')
          const dojoRoot = dojoB && dojoB.parent ? dojoB.parent : envScene.children[0]
          if (dojoRoot) {
            dojoRoot.position.set(-7.07, 1.104, 14.0)
            dojoRoot.rotation.set(0, 0, 0)
            dojoRoot.scale.set(1, 1, 1)
          }

          const embeddedLights: THREE.Object3D[] = []
          envScene.traverse((child) => {
            if ((child as THREE.Light).isLight) {
              embeddedLights.push(child)
              return
            }
            const name = child.name || ''
            if (
              name.startsWith('CubeLight') ||
              name.startsWith('CylinderLight') ||
              name.startsWith('_Colliders')
            ) {
              child.visible = false
              return
            }
            if ((child as THREE.Mesh).isMesh) {
              child.receiveShadow = true
              const isStructure =
                name.includes('Ceiling') ||
                name.includes('Roof') ||
                name.includes('Wall') ||
                name.includes('Floor') ||
                name.includes('Outside')
              child.castShadow = !isStructure
            }
          })

          embeddedLights.forEach((light) => {
            light.parent?.remove(light)
          })

          dojoModel = envScene
          scene.add(dojoModel)
          gridHelper.visible = false
        },
        undefined,
        () => {
          if (!disposed && allowFallback) {
            loadDojo('/environment/dojo.glb', false)
          }
        },
      )
    }
    loadDojo('/media/Dojo.glb', true)

    // 2. Load the animated User Rig GLB
    loader.load(
      url,
      (gltf) => {
        if (disposed) {
          disposeModel(gltf.scene)
          return
        }
        model = gltf.scene
        model.scale.setScalar(avatarScale)

        model.traverse((object) => {
          if ((object as THREE.Mesh).isMesh) {
            const mesh = object as THREE.Mesh
            mesh.castShadow = true
            mesh.receiveShadow = true
            mesh.frustumCulled = false
          }
        })

        scene.add(model)

        if (gltf.animations.length > 0) {
          mixer = new THREE.AnimationMixer(model)
          const clip = gltf.animations[0]
          animation = mixer.clipAction(clip)
          animation.setLoop(THREE.LoopOnce, 1).play()
          animation.clampWhenFinished = true
          animation.paused = true
          mixer.update(0)
        }

        const clipDuration = animation?.getClip().duration ?? 0
        durationCallbackRef.current?.(clipDuration)

        // Frame the 1.75x avatar inside the Dojo
        const box = new THREE.Box3().setFromObject(model)
        const size = box.getSize(new THREE.Vector3())
        const center = box.getCenter(new THREE.Vector3())
        if (Number.isFinite(center.y) && size.y > 0.1) {
          controls.target.set(center.x, Math.max(0.9, center.y), center.z)
          camera.position.set(center.x, Math.max(1.3, center.y), center.z + Math.max(size.y * 1.5, 2.8))
          controls.update()
        }

        setStatus(clipDuration > 0 ? '' : 'The generated 3D rig has no animation.')
      },
      (e) => !disposed && e.total && setStatus(`Loading 3D rig… ${Math.round((e.loaded / e.total) * 100)}%`),
      () => {
        if (disposed) return
        setStatus('Could not load the generated 3D rig.')
        durationCallbackRef.current?.(0)
      },
    )

    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const explicitTime = mocapTimeRef.current
      const time = explicitTime !== undefined ? explicitTime : (video?.current?.currentTime ?? 0)
      if (animation && mixer) {
        const clipDur = animation.getClip().duration || 1
        animation.time = Math.max(0, Math.min(time, clipDur - 0.0005))
        mixer.update(0)
      }
      controls.update()
      renderer.render(scene, camera)
    }
    loop()

    const handleResize = () => {
      const { width, height } = measure()
      renderer.setSize(width, height)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }

    const ro = new ResizeObserver(handleResize)
    ro.observe(el.parentElement ?? el)
    window.addEventListener('resize', handleResize)

    return () => {
      disposed = true
      presetHandlerRef.current = null
      cancelAnimationFrame(raf)
      ro.disconnect()
      window.removeEventListener('resize', handleResize)
      controls.dispose()
      mixer?.stopAllAction()
      if (model) disposeModel(model)
      if (dojoModel) disposeModel(dojoModel)
      gridHelper.geometry.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === el) {
        el.removeChild(renderer.domElement)
      }
    }
  }, [url, video])

  return (
    <div ref={host} className="absolute inset-0 h-full w-full">
      {status && (
        <div className="stencil pointer-events-none absolute inset-0 z-10 grid place-items-center bg-black/50 text-sm text-cream/80">
          {status}
        </div>
      )}
      <div className="absolute bottom-3 right-3 z-10 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => presetHandlerRef.current?.('perspective')}
          className={`ctrl-btn ${activePreset === 'perspective' ? 'active' : ''}`}
          title="3/4 Perspective"
        >
          3/4
        </button>
        <button
          type="button"
          onClick={() => presetHandlerRef.current?.('front')}
          className={`ctrl-btn ${activePreset === 'front' ? 'active' : ''}`}
          title="Front View"
        >
          Front
        </button>
        <button
          type="button"
          onClick={() => presetHandlerRef.current?.('side')}
          className={`ctrl-btn ${activePreset === 'side' ? 'active' : ''}`}
          title="Side View"
        >
          Side
        </button>
        <button
          type="button"
          onClick={() => presetHandlerRef.current?.('upper')}
          className={`ctrl-btn ${activePreset === 'upper' ? 'active' : ''}`}
          title="Upper Body"
        >
          Upper
        </button>
      </div>
    </div>
  )
}
