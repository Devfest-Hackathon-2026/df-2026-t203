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
    this.animationClip = null;

    this.duration = 0;
    this.isPlaying = false;
    this.isLooping = true;
    this.playbackRate = 1.0;

    this.isSeeking = false;
    this.animationFrameId = null;

    this.initScene();
    this.initLights();
    this.initFloor();
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
    this.scene.background = new THREE.Color(0x131722);
    this.scene.fog = new THREE.FogExp2(0x131722, 0.04);

    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 600;

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(0, 1.3, 3.2);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    // Insert the canvas behind the overlay controls (camera presets, video PIP)
    const canvas = this.renderer.domElement;
    canvas.style.position = 'absolute';
    canvas.style.inset = '0';
    this.container.insertBefore(canvas, this.container.firstChild);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 1.0, 0);
    this.controls.maxPolarAngle = Math.PI / 2 + 0.05; // Don't go below floor
    this.controls.minDistance = 0.8;
    this.controls.maxDistance = 15;
    this.controls.update();
  }

  initLights() {
    // Ambient light
    const ambient = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(ambient);

    // Main key light
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
    keyLight.position.set(3, 5, 3);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 15;
    keyLight.shadow.bias = -0.0005;
    this.scene.add(keyLight);

    // Rim / Back light
    const rimLight = new THREE.DirectionalLight(0x6366f1, 1.2);
    rimLight.position.set(-3, 3, -3);
    this.scene.add(rimLight);

    // Fill light
    const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.5);
    fillLight.position.set(0, -1, 3);
    this.scene.add(fillLight);
  }

  initFloor() {
    // Grid floor
    const gridHelper = new THREE.GridHelper(10, 20, 0x6366f1, 0x2b3245);
    gridHelper.position.y = 0;
    this.scene.add(gridHelper);

    // Shadow receiver plane
    const planeGeo = new THREE.PlaneGeometry(30, 30);
    const planeMat = new THREE.ShadowMaterial({ opacity: 0.35 });
    const shadowPlane = new THREE.Mesh(planeGeo, planeMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = -0.005;
    shadowPlane.receiveShadow = true;
    this.scene.add(shadowPlane);
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

  setCameraPreset(preset) {
    if (!this.controls) return;
    const target = this.controls.target;
    switch (preset) {
      case 'front':
        this.camera.position.set(target.x, target.y + 0.1, target.z + 3.0);
        break;
      case 'side':
        this.camera.position.set(target.x + 3.0, target.y + 0.1, target.z);
        break;
      case 'perspective':
        this.camera.position.set(target.x + 2.0, target.y + 0.8, target.z + 2.5);
        break;
      case 'upper':
        this.camera.position.set(target.x, target.y + 0.3, target.z + 1.8);
        this.controls.target.set(target.x, target.y + 0.4, target.z);
        break;
    }
    this.controls.update();
  }

  animate() {
    this.animationFrameId = requestAnimationFrame(this.animate);
    const delta = this.clock.getDelta();

    if (this.mixer && this.isPlaying && !this.isSeeking) {
      this.mixer.update(delta);
      if (this.activeAction && this.duration > 0) {
        this.options.onTimeUpdate(this.activeAction.time % this.duration, this.duration);
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

