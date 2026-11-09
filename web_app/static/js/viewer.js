/**
 * Three.js 3D Viewport and Animation Player for Ninjoucon Mocap Studio
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export class AvatarViewer {
  constructor(canvasContainerId, options = {}) {
    this.container = document.getElementById(canvasContainerId);
    if (!this.container) {
      throw new Error(`Container #${canvasContainerId} not found`);
    }

    this.options = {
      onTimeUpdate: options.onTimeUpdate || (() => {}),
      onLoaded: options.onLoaded || (() => {}),
      onFinished: options.onFinished || (() => {}),
      ...options,
    };

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.clock = new THREE.Clock();

    this.mixer = null;
    this.activeAction = null;
    this.currentModel = null;
    this.environmentModel = null;
    this.gridHelper = null;
    this.animationClip = null;

    this.duration = 0;
    this.isPlaying = false;
    this.isLooping = true;
    this.playbackRate = 1.0;
    this.keyframeBarrierTime = null;

    this.isSeeking = false;
    this.animationFrameId = null;

    this.initScene();
    this.initLights();
    this.initFloor();
    this.loadEnvironment(options.environmentUrl || '/environment/dojo.glb');
    this.animate = this.animate.bind(this);
    this.handleResize = this.handleResize.bind(this);

    window.addEventListener('resize', this.handleResize);
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.handleResize());
      this.resizeObserver.observe(this.container);
    }
    this.animate();
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1612);
    this.scene.fog = new THREE.FogExp2(0x1a1612, 0.015);

    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 600;

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 150);
    this.camera.position.set(0, 1.3, 3.2);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;

    // Insert the canvas behind the overlay controls (camera presets, video PIP)
    const canvas = this.renderer.domElement;
    canvas.style.position = 'absolute';
    canvas.style.inset = '0';
    this.container.insertBefore(canvas, this.container.firstChild);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 1.0, 0);
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // Keep camera above the dojo wooden floor
    this.controls.minDistance = 0.8;
    this.controls.maxDistance = 12.0; // Keep camera inside the dojo walls
    this.controls.update();
  }

  initLights() {
    // Warm interior hemisphere & ambient light for the dojo
    const hemiLight = new THREE.HemisphereLight(0xfff4e0, 0x4a3525, 1.1);
    hemiLight.position.set(0, 4.5, 0);
    this.scene.add(hemiLight);

    const ambient = new THREE.AmbientLight(0xffeedd, 0.6);
    this.scene.add(ambient);

    // Main key light positioned inside the dojo below ceiling height
    const keyLight = new THREE.DirectionalLight(0xfff7e8, 1.6);
    keyLight.position.set(2.5, 4.2, 3.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 15;
    keyLight.shadow.camera.left = -4;
    keyLight.shadow.camera.right = 4;
    keyLight.shadow.camera.top = 4;
    keyLight.shadow.camera.bottom = -4;
    keyLight.shadow.bias = -0.0005;
    this.scene.add(keyLight);

    // Warm lantern / shrine back-rim light inside the dojo
    const rimLight = new THREE.PointLight(0xffb366, 6.0, 22, 1.5);
    rimLight.position.set(0, 3.2, -4.5);
    this.scene.add(rimLight);

    // Soft front/side fill light
    const fillLight = new THREE.DirectionalLight(0xdbeafe, 0.5);
    fillLight.position.set(-3, 2.5, 4);
    this.scene.add(fillLight);
  }

  initFloor() {
    // Fallback grid floor (hidden automatically once Dojo.glb loads)
    this.gridHelper = new THREE.GridHelper(10, 20, 0x6366f1, 0x2b3245);
    this.gridHelper.position.y = 0;
    this.scene.add(this.gridHelper);

    // Subtle contact shadow receiver plane right at floor level (y = 0)
    const planeGeo = new THREE.PlaneGeometry(30, 30);
    const planeMat = new THREE.ShadowMaterial({ opacity: 0.3 });
    const shadowPlane = new THREE.Mesh(planeGeo, planeMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = 0.002;
    shadowPlane.receiveShadow = true;
    this.scene.add(shadowPlane);
  }

  async loadEnvironment(url) {
    if (!url) return null;
    const loader = new GLTFLoader();
    return new Promise((resolve) => {
      loader.load(
        url,
        (gltf) => {
          if (this.environmentModel) {
            this.scene.remove(this.environmentModel);
          }

          const envScene = gltf.scene;

          // Align the Dojo building so its interior wooden floor is at y = 0
          // and the avatar at (0, 0, 0) stands centered in front of the main shrine wall.
          const dojoB = envScene.getObjectByName('Dojo_B');
          const dojoRoot = (dojoB && dojoB.parent) ? dojoB.parent : envScene.children[0];
          if (dojoRoot) {
            dojoRoot.position.set(-7.07, 1.104, 14.0);
            dojoRoot.rotation.set(0, 0, 0);
            dojoRoot.scale.set(1, 1, 1);
          }

          const embeddedLights = [];
          envScene.traverse((child) => {
            if (child.isLight) {
              embeddedLights.push(child);
              return;
            }
            const name = child.name || '';
            if (
              name.startsWith('CubeLight') ||
              name.startsWith('CylinderLight') ||
              name.startsWith('_Colliders')
            ) {
              child.visible = false;
              return;
            }
            if (child.isMesh) {
              child.receiveShadow = true;
              // Allow interior props to cast shadows without the roof/ceiling blocking overhead light
              const isStructure =
                name.includes('Ceiling') ||
                name.includes('Roof') ||
                name.includes('Wall') ||
                name.includes('Floor') ||
                name.includes('Outside');
              child.castShadow = !isStructure;
            }
          });

          // Remove the 30 raw KHR_lights_punctual lights to prevent shader uniform bloat and overexposure
          embeddedLights.forEach((light) => {
            if (light.parent) light.parent.remove(light);
          });

          this.environmentModel = envScene;
          this.scene.add(this.environmentModel);

          if (this.gridHelper) {
            this.gridHelper.visible = false;
          }

          resolve(gltf);
        },
        undefined,
        (err) => {
          console.warn('[viewer] Failed to load environment GLB:', err);
          resolve(null);
        }
      );
    });
  }

  handleResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  async loadModel(url) {
    if (this.currentModel) {
      this.scene.remove(this.currentModel);
      this.currentModel = null;
    }
    if (this.mixer) {
      this.mixer.stopAllAction();
      this.mixer = null;
    }

    const loader = new GLTFLoader();
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (gltf) => {
          this.currentModel = gltf.scene;
          this.currentModel.scale.setScalar(1.75);

          // Enable shadows on all meshes
          this.currentModel.traverse((child) => {
            if (child.isMesh) {
              child.castShadow = true;
              child.receiveShadow = true;
            }
          });

          this.scene.add(this.currentModel);

          // Find animations
          const animations = gltf.animations || [];
          if (animations.length > 0) {
            this.mixer = new THREE.AnimationMixer(this.currentModel);
            this.animationClip = animations[0];
            this.duration = this.animationClip.duration;
            this.activeAction = this.mixer.clipAction(this.animationClip);
            this.activeAction.setLoop(this.isLooping ? THREE.LoopRepeat : THREE.LoopOnce);
            this.activeAction.clampWhenFinished = !this.isLooping;
            // Start paused on the first frame; the user presses Play.
            this.activeAction.play();
            this.activeAction.paused = true;
            this.mixer.update(0);
            this.isPlaying = false;

            this.mixer.addEventListener('finished', () => {
              this.isPlaying = false;
              this.options.onFinished();
            });

            this.options.onLoaded(this.duration);
          } else {
            console.warn('[viewer] GLB loaded but has no animation tracks');
            this.duration = 0;
            this.options.onLoaded(0);
          }

          // Center model / camera framing
          this.handleResize();
          this.frameModel();
          resolve(gltf);
        },
        undefined,
        (err) => {
          console.error('[viewer] Failed to load GLB:', err);
          reject(err);
        }
      );
    });
  }

  frameModel() {
    if (!this.currentModel) return;
    const box = new THREE.Box3().setFromObject(this.currentModel);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    this.controls.target.set(center.x, Math.max(0.9, center.y), center.z);
    this.camera.position.set(center.x, Math.max(1.3, center.y), center.z + Math.max(size.y * 1.5, 2.8));
    this.controls.update();
  }

  play() {
    if (this.activeAction) {
      if (!this.isPlaying && this.activeAction.time >= this.duration) {
        this.activeAction.reset();
      }
      this.activeAction.paused = false;
      this.isPlaying = true;
    }
  }

  pause() {
    if (this.activeAction) {
      this.activeAction.paused = true;
      this.isPlaying = false;
    }
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
    return this.isPlaying;
  }

  reset() {
    if (this.activeAction) {
      this.activeAction.reset();
      this.activeAction.paused = true;
      this.isPlaying = false;
      if (this.mixer) this.mixer.update(0);
      this.options.onTimeUpdate(0, this.duration);
    }
  }

  seek(timeInSeconds) {
    if (!this.activeAction || !this.mixer) return;
    const clampedTime = Math.max(0, Math.min(this.duration, timeInSeconds));
    this.activeAction.time = clampedTime;
    this.mixer.update(0);
    this.options.onTimeUpdate(clampedTime, this.duration);
  }

  setSpeed(speed) {
    this.playbackRate = speed;
    if (this.mixer) {
      this.mixer.timeScale = speed;
    }
  }

  setLoop(isLoop) {
    this.isLooping = isLoop;
    if (this.activeAction) {
      this.activeAction.setLoop(isLoop ? THREE.LoopRepeat : THREE.LoopOnce);
      this.activeAction.clampWhenFinished = !isLoop;
    }
  }

  setKeyframeBarrier(targetTime) {
    if (targetTime === null || targetTime === undefined || !Number.isFinite(targetTime)) {
      this.keyframeBarrierTime = null;
    } else {
      this.keyframeBarrierTime = Math.max(0, Number(targetTime));
    }
  }

  setCameraPreset(preset) {
    if (!this.controls) return;
    const s = 1.75;
    const baseY = 0.9 * s;
    this.controls.target.set(0, baseY, 0);
    const target = this.controls.target;
    switch (preset) {
      case 'front':
        this.camera.position.set(target.x, target.y + 0.1 * s, target.z + 3.0 * s);
        break;
      case 'side':
        this.camera.position.set(target.x + 3.0 * s, target.y + 0.1 * s, target.z);
        break;
      case 'perspective':
        this.camera.position.set(target.x + 2.0 * s, target.y + 0.8 * s, target.z + 2.5 * s);
        break;
      case 'upper':
        this.controls.target.set(0, (0.9 + 0.4) * s, 0);
        this.camera.position.set(0, (0.9 + 0.3) * s, 1.8 * s);
        break;
    }
    this.controls.update();
  }

  animate() {
    this.animationFrameId = requestAnimationFrame(this.animate);
    const delta = this.clock.getDelta();

    if (this.mixer && this.isPlaying && !this.isSeeking) {
      if (
        this.activeAction &&
        this.duration > 0 &&
        this.keyframeBarrierTime !== null
      ) {
        const nextTime = this.activeAction.time + delta * this.playbackRate;
        if (nextTime >= this.keyframeBarrierTime) {
          const clampedBarrier = Math.max(
            0,
            Math.min(this.duration - 0.0005, this.keyframeBarrierTime)
          );
          this.activeAction.time = clampedBarrier;
          this.mixer.update(0);
          this.options.onTimeUpdate(this.keyframeBarrierTime, this.duration);
        } else {
          this.mixer.update(delta);
          this.options.onTimeUpdate(
            this.activeAction.time % this.duration,
            this.duration
          );
        }
      } else {
        this.mixer.update(delta);
        if (this.activeAction && this.duration > 0) {
          this.options.onTimeUpdate(
            this.activeAction.time % this.duration,
            this.duration
          );
        }
      }
    }

    if (this.controls) {
      this.controls.update();
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  destroy() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    window.removeEventListener('resize', this.handleResize);
    if (this.renderer && this.renderer.domElement) {
      this.renderer.dispose();
      if (this.renderer.domElement.parentElement) {
        this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
      }
    }
  }
}

