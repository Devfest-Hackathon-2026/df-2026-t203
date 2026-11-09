/**
 * Main Web Application Controller for Ninjoucon Motion Capture & Copy-Training Studio
 */

import { AvatarViewer } from './viewer.js?v=20261110_1';

class App {
  constructor() {
    this.videos = [];
    this.miscRecordings = [];
    this.expandedVideos = new Set(['misc_mocap']); // Set of video_ids (plus 'misc_mocap') whose subdirectories are expanded
    this.selectedVideo = null;
    this.selectedRecording = null;
    this.currentMode = 'add-video'; // 'add-video', 'teacher-video', 'test-mocap', 'unprocessed', 'processed'

    this.pollInterval = null;
    this.keyposePollInterval = null;

    // Copy-Training WebSocket & rAF state
    this.trainingWs = null;
    this.playUntilRaf = null;
    this.checkpointEditorOpen = false;

    // Standalone Test Motion Capture Camera & MediaRecorder
    this.mediaStream = null;
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.recordingStartTime = null;
    this.timerInterval = null;

    // Three.js Viewer
    this.viewer = null;

    this.initElements();
    this.bindEvents();
    this.init();
  }

  initElements() {
    // Sidebar & Navigation
    this.sidebar = document.querySelector('.sidebar');
    this.btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
    this.recordingsListEl = document.getElementById('recordings-list');
    this.btnAddVideo = document.getElementById('btn-add-video');
    this.btnTestMocap = document.getElementById('btn-test-mocap');

    // Main Views
    this.addVideoView = document.getElementById('add-video-view');
    this.teacherVideoView = document.getElementById('teacher-video-view');
    this.testMocapView = document.getElementById('test-mocap-view');
    this.unprocessedView = document.getElementById('unprocessed-view');
    this.processedView = document.getElementById('processed-view');

    // Add Teacher Video View Elements
    this.youtubeUrlInput = document.getElementById('youtube-url-input');
    this.youtubeNameInput = document.getElementById('youtube-name-input');
    this.btnLoadYoutube = document.getElementById('btn-load-youtube');
    this.btnLoadYoutubeText = document.getElementById('btn-load-youtube-text');
    this.uploadVideoNameInput = document.getElementById('upload-video-name-input');
    this.uploadDropzone = document.getElementById('upload-dropzone');
    this.fileUploadInput = document.getElementById('file-upload-input');

    // Test Motion Capture View Elements (Standalone Misc Recordings)
    this.cameraVideo = document.getElementById('camera-video');
    this.cameraSelect = document.getElementById('camera-select');
    this.btnStartRecord = document.getElementById('btn-start-record');
    this.btnStopRecord = document.getElementById('btn-stop-record');
    this.recIndicator = document.getElementById('rec-indicator');
    this.recTimer = document.getElementById('rec-timer');
    this.mocapTestDropzone = document.getElementById('mocap-test-dropzone');
    this.mocapTestFileInput = document.getElementById('mocap-test-file-input');

    // Teacher Video View + Integrated YouTube Extension Elements
    this.teacherVideoTitle = document.getElementById('teacher-video-title');
    this.teacherVideoSubtitle = document.getElementById('teacher-video-subtitle');
    this.teacherKeyposeBadge = document.getElementById('teacher-keypose-badge');
    this.teacherPlayerContainer = document.getElementById('teacher-player-container');
    this.teacherVideo = document.getElementById('teacher-video');

    // Extension Checkpoint Controls
    this.segmentBtnContainer = document.getElementById('segment-btn-container');
    this.btnSaveCheckpoint = document.getElementById('btn-save-checkpoint');
    this.btnClearCheckpoint = document.getElementById('btn-clear-checkpoint');
    this.checkpointSavedPill = document.getElementById('checkpoint-saved-pill');
    this.checkpointSavedText = document.getElementById('checkpoint-saved-text');
    this.btnEditCheckpoint = document.getElementById('btn-edit-checkpoint');

    this.segmentOverlayContainer = document.getElementById('segment-overlay-container');
    this.checkpointGhostContainer = document.getElementById('checkpoint-ghost-container');
    this.checkpointPrimaryBar = document.getElementById('checkpoint-primary-bar');
    this.checkpointTimeLabel = document.getElementById('checkpoint-time-label');
    this.checkpointSlider = document.getElementById('checkpoint-slider');

    // Extension Webcam Overlay & Side-by-Side Controls
    this.webcamOverlay = document.getElementById('motionlearn-webcam-overlay');
    this.webcamOverlayLabel = document.getElementById('webcam-overlay-label');
    this.webcamOpacityControl = document.getElementById('webcam-opacity-control');
    this.overlayOpacitySlider = document.getElementById('overlay-opacity-slider');
    this.btnToggleTrainingLayout = document.getElementById('btn-toggle-training-layout');
    this.btnExpandOverlay = document.getElementById('btn-expand-overlay');
    this.opencvOverlayImg = document.getElementById('opencv-overlay-img');
    this.opencvOverlayStatus = document.getElementById('opencv-overlay-status');
    this.trainingSideBySide = false;

    // Teacher Process & Copy-Training Cards
    this.teacherProcessCard = document.getElementById('teacher-process-card');
    this.btnProcessKeyposes = document.getElementById('btn-process-keyposes');
    this.btnProcessKeyposesText = document.getElementById('btn-process-keyposes-text');
    this.teacherKeyposeLogs = document.getElementById('teacher-keypose-logs');

    this.teacherTrainingCard = document.getElementById('teacher-training-card');
    this.trainingHeadline = document.getElementById('training-headline');
    this.trainingSubtext = document.getElementById('training-subtext');
    this.btnStartTraining = document.getElementById('btn-start-training');
    this.btnStartTrainingText = document.getElementById('btn-start-training-text');
    this.btnTrainingLayoutBar = document.getElementById('btn-training-layout-bar');
    this.btnSkipPose = document.getElementById('btn-skip-pose');
    this.btnCheckpointContinue = document.getElementById('btn-checkpoint-continue');
    this.btnCheckpointRetry = document.getElementById('btn-checkpoint-retry');
    this.btnStopTraining = document.getElementById('btn-stop-training');

    this.trainingTelemetryBar = document.getElementById('training-telemetry-bar');
    this.telemetryPose = document.getElementById('telemetry-pose');
    this.telemetryState = document.getElementById('telemetry-state');
    this.telemetryError = document.getElementById('telemetry-error');
    this.telemetryHold = document.getElementById('telemetry-hold');
    this.telemetryDtw = document.getElementById('telemetry-dtw');

    // Unprocessed Recording View Elements
    this.unprocessedTitle = document.getElementById('unprocessed-title');
    this.unprocessedDate = document.getElementById('unprocessed-date');
    this.unprocessedVideo = document.getElementById('unprocessed-video');
    this.downloadRawUserVideo = document.getElementById('download-raw-user-video');
    this.btnProcess = document.getElementById('btn-process');
    this.btnProcessText = document.getElementById('btn-process-text');
    this.processStageText = document.getElementById('process-stage-text');
    this.consoleLogs = document.getElementById('console-logs');

    // Processed 3D View Elements
    this.processedTitle = document.getElementById('processed-title');
    this.processedDate = document.getElementById('processed-date');
    this.threeContainer = document.getElementById('three-canvas-container');
    this.btnPlay = document.getElementById('btn-play');
    this.btnPlayIcon = document.getElementById('btn-play-icon');
    this.btnReset = document.getElementById('btn-reset');
    this.btnPrevCheckpoint = document.getElementById('btn-prev-checkpoint');
    this.btnNextCheckpoint = document.getElementById('btn-next-checkpoint');
    this.btnLoop = document.getElementById('btn-loop');
    this.timeLabel = document.getElementById('time-label');
    this.timelineScrubber = document.getElementById('timeline-scrubber');
    this.mocapKeyposeDatalist = document.getElementById('mocap-keypose-datalist');
    this.keyposeStepBadge = document.getElementById('keypose-step-badge');
    this.speedSelect = document.getElementById('speed-select');
    this.btnTogglePip = document.getElementById('btn-toggle-pip');
    this.videoPanel = document.getElementById('video-panel');
    this.videoPanelLabel = document.getElementById('video-panel-label');
    this.pipKeyposeStatus = document.getElementById('pip-keypose-status');
    this.pipVideo = document.getElementById('pip-video');
    this.downloadUserMp4 = document.getElementById('download-user-mp4');
    this.downloadGlb = document.getElementById('download-glb');
    this.downloadBlend = document.getElementById('download-blend');
    this.downloadFbx = document.getElementById('download-fbx');
    this.isTeacherSyncedRecording = false;
    this.processedKeyposeSync = [];
    this.pipSyncState = { activeIdx: -1, phase: 'idle', lastSeekTarget: -1 };
  }

  bindEvents() {
    // Collapsible Sidebar
    if (this.btnToggleSidebar && this.sidebar) {
      this.btnToggleSidebar.addEventListener('click', () => {
        this.sidebar.classList.toggle('collapsed');
        setTimeout(() => {
          if (this.viewer) this.viewer.handleResize();
        }, 220);
      });
    }

    // Add Video Button
    this.btnAddVideo.addEventListener('click', () => {
      this.selectedVideo = null;
      this.selectedRecording = null;
      this.renderSidebar();
      this.switchMode('add-video');
    });

    // Test Motion Capture Button (Standalone Misc Recordings)
    this.btnTestMocap.addEventListener('click', () => {
      this.selectedVideo = null;
      this.selectedRecording = null;
      this.expandedVideos.add('misc_mocap');
      this.renderSidebar();
      this.switchMode('test-mocap');
    });

    // Standalone Test Motion Capture Recording Events
    this.btnStartRecord.addEventListener('click', () => this.startMiscRecording());
    this.btnStopRecord.addEventListener('click', () => this.stopMiscRecording());
    if (this.cameraSelect) {
      this.cameraSelect.addEventListener('change', () => this.initCamera(this.cameraSelect.value));
    }

    if (this.mocapTestDropzone && this.mocapTestFileInput) {
      this.mocapTestDropzone.addEventListener('click', () => this.mocapTestFileInput.click());
      this.mocapTestFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.uploadMiscFile(e.target.files[0]);
        }
      });
      this.mocapTestDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        this.mocapTestDropzone.style.borderColor = 'var(--accent)';
      });
      this.mocapTestDropzone.addEventListener('dragleave', () => {
        this.mocapTestDropzone.style.borderColor = 'var(--border)';
      });
      this.mocapTestDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        this.mocapTestDropzone.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files.length > 0) {
          this.uploadMiscFile(e.dataTransfer.files[0]);
        }
      });
    }

    // Option 1: Load YouTube Video
    this.btnLoadYoutube.addEventListener('click', () => this.handleAddYoutubeVideo());
    this.youtubeUrlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.handleAddYoutubeVideo();
    });

    // Option 2: Upload MP4 Teacher Video File
    this.uploadDropzone.addEventListener('click', () => this.fileUploadInput.click());
    this.fileUploadInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        this.handleUploadTeacherVideo(e.target.files[0]);
      }
    });

    this.uploadDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.uploadDropzone.style.borderColor = 'var(--accent)';
    });
    this.uploadDropzone.addEventListener('dragleave', () => {
      this.uploadDropzone.style.borderColor = 'var(--border)';
    });
    this.uploadDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadDropzone.style.borderColor = 'var(--border)';
      if (e.dataTransfer.files.length > 0) {
        this.handleUploadTeacherVideo(e.dataTransfer.files[0]);
      }
    });

    // Teacher Video Player Metadata Loaded -> Configure Checkpoint Slider & Keypose Ticks
    this.teacherVideo.addEventListener('loadedmetadata', () => {
      this.refreshCheckpointTimelineUI();
    });

    // Checkpoint Slider Input (Matches Content.js updateSegmentMarkers)
    this.checkpointSlider.addEventListener('input', () => {
      const val = parseFloat(this.checkpointSlider.value || '0');
      this.teacherVideo.currentTime = val;
      this.updateSegmentMarkers(val);
    });

    // Save / Clear / Edit Checkpoint Buttons
    this.btnSaveCheckpoint.addEventListener('click', () => {
      const val = parseFloat(this.checkpointSlider.value || '0');
      this.saveVideoCheckpoint(val > 0 ? val : null);
    });
    this.btnClearCheckpoint.addEventListener('click', () => {
      this.saveVideoCheckpoint(null);
    });
    this.btnEditCheckpoint.addEventListener('click', () => {
      this.checkpointEditorOpen = true;
      this.refreshCheckpointTimelineUI();
    });

    // Expand / Collapse Webcam Overlay inside Player (Full Translucent Overlay vs Corner PIP)
    this.btnExpandOverlay.addEventListener('click', () => {
      if (this.trainingSideBySide) {
        this.toggleTrainingLayout(false);
      }
      const isExpanded = this.webcamOverlay.classList.toggle('expanded');
      this.btnExpandOverlay.textContent = isExpanded ? '🗗 Corner PIP' : '⛶ Full Overlay';
    });

    // Opacity slider for Translucent Webcam Overlay
    if (this.overlayOpacitySlider) {
      this.overlayOpacitySlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value || '0.45');
        this.webcamOverlay.style.setProperty('--webcam-overlay-opacity', String(val));
      });
    }

    // Toggle between Translucent Overlay and Side-by-Side
    if (this.btnToggleTrainingLayout) {
      this.btnToggleTrainingLayout.addEventListener('click', () => this.toggleTrainingLayout());
    }
    if (this.btnTrainingLayoutBar) {
      this.btnTrainingLayoutBar.addEventListener('click', () => this.toggleTrainingLayout());
    }

    // Process Teacher Video into Keyposes
    this.btnProcessKeyposes.addEventListener('click', () => this.startTeacherKeyposeProcessing());

    // Copy-Training & Clean Recording Controls
    this.btnStartTraining.addEventListener('click', () => this.startTrainingSession());
    this.btnStopTraining.addEventListener('click', () => this.stopTrainingSession());
    this.btnSkipPose.addEventListener('click', () => this.sendSkipPose());
    this.btnCheckpointContinue.addEventListener('click', () => this.sendCheckpointAction('Thumb_Up'));
    this.btnCheckpointRetry.addEventListener('click', () => this.sendCheckpointAction('Thumb_Down'));

    // Global 'S' shortcut to skip pose during active training (from Content.js)
    window.addEventListener('keydown', (e) => {
      if ((e.key === 's' || e.key === 'S') && this.trainingWs && this.trainingWs.readyState === WebSocket.OPEN) {
        if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
          return;
        }
        this.sendSkipPose();
      }
    });

    // Unprocessed Recording -> Run Motion Capture Pipeline
    this.btnProcess.addEventListener('click', () => this.startProcessing());

    // Processed 3D Player Controls
    this.btnPlay.addEventListener('click', () => this.togglePlayback());
    this.btnReset.addEventListener('click', () => {
      if (this.viewer) this.viewer.reset();
      this.pipSyncState = { activeIdx: -1, phase: 'idle', lastSeekTarget: -1 };
      this.syncPipVideoWithMocap(0, false, true);
      this.updatePlayButtonState(false);
    });

    if (this.btnPrevCheckpoint) {
      this.btnPrevCheckpoint.addEventListener('click', () => this.stepProcessedCheckpoint(-1));
    }
    if (this.btnNextCheckpoint) {
      this.btnNextCheckpoint.addEventListener('click', () => this.stepProcessedCheckpoint(1));
    }

    this.btnLoop.addEventListener('click', () => {
      const isLoop = !this.btnLoop.classList.contains('active');
      this.btnLoop.classList.toggle('active', isLoop);
      if (this.viewer) this.viewer.setLoop(isLoop);
      if (this.pipVideo) {
        this.pipVideo.loop = !this.isTeacherSyncedRecording && isLoop;
      }
    });

    const startScrub = () => {
      if (this.viewer) this.viewer.isSeeking = true;
      if (this.pipVideo && !this.pipVideo.paused) {
        this.pipVideo.pause();
      }
    };

    const endScrub = () => {
      if (this.viewer) {
        this.viewer.isSeeking = false;
        const currentMocapTime = parseFloat(this.timelineScrubber.value || '0');
        this.pipSyncState = { activeIdx: -1, phase: 'idle', lastSeekTarget: -1 };
        this.syncPipVideoWithMocap(currentMocapTime, Boolean(this.viewer.isPlaying), true);
      }
    };

    this.timelineScrubber.addEventListener('pointerdown', startScrub);
    this.timelineScrubber.addEventListener('pointerup', endScrub);
    this.timelineScrubber.addEventListener('change', endScrub);

    this.timelineScrubber.addEventListener('input', (e) => {
      let time = parseFloat(e.target.value);
      if (this.isTeacherSyncedRecording && this.processedKeyposeSync.length > 0) {
        time = this.snapMocapTimeToKeypose(time);
        this.timelineScrubber.value = time;
      }
      if (this.viewer) this.viewer.seek(time);
      this.syncPipVideoWithMocap(time, false, false);
    });

    this.speedSelect.addEventListener('change', (e) => {
      const speed = parseFloat(e.target.value);
      if (this.viewer) this.viewer.setSpeed(speed);
      if (this.pipVideo) {
        this.pipVideo.playbackRate = speed;
      }
    });

    if (this.pipVideo) {
      this.pipVideo.addEventListener('loadedmetadata', () => {
        if (this.isTeacherSyncedRecording && this.viewer && this.viewer.duration) {
          this.initProcessedKeyposeSync(this.viewer.duration);
        }
        const currentTime = parseFloat(this.timelineScrubber.value || '0');
        this.pipSyncState = { activeIdx: -1, phase: 'idle', lastSeekTarget: -1 };
        this.syncPipVideoWithMocap(currentTime, Boolean(this.viewer && this.viewer.isPlaying), true);
      });
    }

    if (this.btnTogglePip && this.videoPanel) {
      this.btnTogglePip.addEventListener('click', () => {
        const isVisible = this.videoPanel.style.display !== 'none';
        const nextVisible = !isVisible;
        this.videoPanel.style.display = nextVisible ? 'flex' : 'none';
        this.btnTogglePip.classList.toggle('active', nextVisible);

        if (nextVisible && this.pipVideo) {
          const currentTime = parseFloat(this.timelineScrubber.value || '0');
          this.pipVideo.loop = !this.isTeacherSyncedRecording && this.btnLoop.classList.contains('active');
          this.syncPipVideoWithMocap(currentTime, Boolean(this.viewer && this.viewer.isPlaying));
        } else if (this.pipVideo) {
          this.pipVideo.pause();
        }

        if (this.viewer) {
          requestAnimationFrame(() => this.viewer.handleResize());
        }
      });
    }

    // Camera preset buttons
    document.querySelectorAll('[data-camera]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const preset = e.currentTarget.dataset.camera;
        if (this.viewer) this.viewer.setCameraPreset(preset);
      });
    });
  }

  async init() {
    await this.fetchVideos();
    if (this.videos.length > 0) {
      this.expandedVideos.add(this.videos[0].id);
      this.selectTeacherVideo(this.videos[0]);
    } else if (this.miscRecordings.length > 0) {
      this.expandedVideos.add('misc_mocap');
      this.selectRecording(this.miscRecordings[0], null);
    } else {
      this.switchMode('add-video');
    }
  }

  // -------------------------------------------------------------
  // Sidebar Data & Subdirectory Tree Rendering
  // -------------------------------------------------------------
  async fetchVideos() {
    try {
      const [vRes, rRes] = await Promise.all([
        fetch('/api/videos'),
        fetch('/api/recordings'),
      ]);
      this.videos = await vRes.json();
      const allRecordings = await rRes.json();
      this.miscRecordings = allRecordings.filter((r) => r.video_id === 'misc_mocap');
      this.renderSidebar();
    } catch (err) {
      console.error('Failed to load videos and recordings:', err);
    }
  }

  renderSidebar() {
    this.recordingsListEl.innerHTML = '';

    // 1. Render Teacher Video folders
    this.videos.forEach((video) => {
      const group = document.createElement('div');
      group.className = 'video-group';

      const isExpanded = this.expandedVideos.has(video.id);
      const isVideoSelected =
        this.currentMode === 'teacher-video' &&
        this.selectedVideo &&
        this.selectedVideo.id === video.id;

      const recordings = video.recordings || [];

      let kpBadge = '';
      if (video.status === 'processing') {
        kpBadge = `<span class="badge badge-processing"><span class="spinner"></span> Keyposes</span>`;
      } else if (video.is_processed) {
        kpBadge = `<span class="badge badge-processed">✓ ${video.keypose_count || 0} Keyposes</span>`;
      } else if (video.status === 'error') {
        kpBadge = `<span class="badge badge-error">⚠ Error</span>`;
      } else {
        kpBadge = `<span class="badge badge-unprocessed">⏳ Unprocessed</span>`;
      }

      const header = document.createElement('div');
      header.className = `video-group-header${isVideoSelected ? ' active' : ''}`;
      header.innerHTML = `
        <div class="video-header-top">
          <div class="video-header-left">
            <button class="expander-btn" type="button" title="${isExpanded ? 'Collapse recordings' : 'Expand recordings'}">
              ${isExpanded ? '▾' : '▸'}
            </button>
            <span class="video-folder-name" title="${this.escapeHtml(video.name)}">🎬 ${this.escapeHtml(video.name)}</span>
          </div>
          ${kpBadge}
        </div>
        <div class="item-bottom">
          <div class="item-date">${recordings.length} recording${recordings.length === 1 ? '' : 's'} • ${video.created_at}</div>
          <button class="delete-btn video-delete-btn" type="button" title="Delete Video & Recordings">🗑</button>
        </div>
      `;

      const expanderBtn = header.querySelector('.expander-btn');
      expanderBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.expandedVideos.has(video.id)) {
          this.expandedVideos.delete(video.id);
        } else {
          this.expandedVideos.add(video.id);
        }
        this.renderSidebar();
      });

      const delVideoBtn = header.querySelector('.video-delete-btn');
      delVideoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteTeacherVideo(video);
      });

      header.addEventListener('click', () => {
        this.expandedVideos.add(video.id);
        this.selectTeacherVideo(video);
      });

      group.appendChild(header);

      const sublist = document.createElement('div');
      sublist.className = `video-recordings-sublist${isExpanded ? '' : ' collapsed'}`;

      if (!video.is_processed) {
        const lockedEl = document.createElement('div');
        lockedEl.className = 'sublist-empty';
        lockedEl.textContent = '🔒 Process video first to take recordings';
        sublist.appendChild(lockedEl);
      } else if (recordings.length === 0) {
        const emptyEl = document.createElement('div');
        emptyEl.className = 'sublist-empty';
        emptyEl.textContent = 'No recordings yet — click Record & Copy Video!';
        sublist.appendChild(emptyEl);
      } else {
        recordings.forEach((rec) => {
          sublist.appendChild(this.createRecordingItemEl(rec, video));
        });
      }

      group.appendChild(sublist);
      this.recordingsListEl.appendChild(group);
    });

    // 2. Render "Test Motion Capture (Misc.)" folder for single misc. recordings
    if (this.miscRecordings.length > 0 || this.currentMode === 'test-mocap') {
      const miscGroup = document.createElement('div');
      miscGroup.className = 'video-group';

      const isExpanded = this.expandedVideos.has('misc_mocap');
      const isMiscSelected = this.currentMode === 'test-mocap';

      const miscHeader = document.createElement('div');
      miscHeader.className = `video-group-header${isMiscSelected ? ' active' : ''}`;
      miscHeader.innerHTML = `
        <div class="video-header-top">
          <div class="video-header-left">
            <button class="expander-btn" type="button" title="${isExpanded ? 'Collapse misc recordings' : 'Expand misc recordings'}">
              ${isExpanded ? '▾' : '▸'}
            </button>
            <span class="video-folder-name" title="Test Motion Capture (Misc.)">🧪 Test Motion Capture (Misc.)</span>
          </div>
          <span class="badge badge-processed">Standalone</span>
        </div>
        <div class="item-bottom">
          <div class="item-date">${this.miscRecordings.length} misc recording${this.miscRecordings.length === 1 ? '' : 's'}</div>
        </div>
      `;

      const miscExpander = miscHeader.querySelector('.expander-btn');
      miscExpander.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.expandedVideos.has('misc_mocap')) {
          this.expandedVideos.delete('misc_mocap');
        } else {
          this.expandedVideos.add('misc_mocap');
        }
        this.renderSidebar();
      });

      miscHeader.addEventListener('click', () => {
        this.selectedVideo = null;
        this.selectedRecording = null;
        this.expandedVideos.add('misc_mocap');
        this.renderSidebar();
        this.switchMode('test-mocap');
      });

      miscGroup.appendChild(miscHeader);

      const miscSublist = document.createElement('div');
      miscSublist.className = `video-recordings-sublist${isExpanded ? '' : ' collapsed'}`;

      if (this.miscRecordings.length === 0) {
        const emptyEl = document.createElement('div');
        emptyEl.className = 'sublist-empty';
        emptyEl.textContent = 'No misc recordings yet — record or upload one!';
        miscSublist.appendChild(emptyEl);
      } else {
        this.miscRecordings.forEach((rec) => {
          miscSublist.appendChild(this.createRecordingItemEl(rec, null));
        });
      }

      miscGroup.appendChild(miscSublist);
      this.recordingsListEl.appendChild(miscGroup);
    }

    if (this.videos.length === 0 && this.miscRecordings.length === 0 && this.currentMode !== 'test-mocap') {
      this.recordingsListEl.innerHTML = `
        <div class="empty-state">
          No videos or recordings yet.<br />Click <strong>"+ Add Video"</strong> or <strong>"Test Motion Capture"</strong> above!
        </div>
      `;
    }
  }

  createRecordingItemEl(rec, parentVideo) {
    const item = document.createElement('div');
    item.className = 'recording-item';
    if (
      (this.currentMode === 'unprocessed' || this.currentMode === 'processed') &&
      this.selectedRecording &&
      this.selectedRecording.id === rec.id
    ) {
      item.classList.add('active');
    }

    let badgeHtml = '';
    if (rec.status === 'processing') {
      badgeHtml = `<span class="badge badge-processing"><span class="spinner"></span> Processing</span>`;
    } else if (rec.is_processed) {
      badgeHtml = `<span class="badge badge-processed">✓ Processed</span>`;
    } else if (rec.status === 'error') {
      badgeHtml = `<span class="badge badge-error">⚠ Error</span>`;
    } else {
      badgeHtml = `<span class="badge badge-unprocessed">⏳ Unprocessed</span>`;
    }

    item.innerHTML = `
      <div class="item-top">
        <span class="item-no">Recording #${rec.recording_no}</span>
        ${badgeHtml}
      </div>
      <div class="item-bottom">
        <div class="item-date">${rec.created_at}</div>
        <button class="delete-btn rec-delete-btn" type="button" title="Delete Recording #${rec.recording_no}">🗑</button>
      </div>
    `;

    const delRecBtn = item.querySelector('.rec-delete-btn');
    delRecBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.deleteRecording(rec);
    });

    item.addEventListener('click', () => this.selectRecording(rec, parentVideo));
    return item;
  }

  // -------------------------------------------------------------
  // Standalone Test Motion Capture (Camera & Misc Upload)
  // -------------------------------------------------------------
  async initCamera(deviceId = null) {
    try {
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((track) => track.stop());
      }

      const constraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId } }
          : { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false,
      };

      this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      if (this.cameraVideo) {
        this.cameraVideo.srcObject = this.mediaStream;
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((d) => d.kind === 'videoinput');
      if (this.cameraSelect && videoDevices.length > 1) {
        this.cameraSelect.innerHTML = videoDevices
          .map((d, i) => `<option value="${d.deviceId}">${d.label || `Camera ${i + 1}`}</option>`)
          .join('');
        this.cameraSelect.style.display = 'block';
      }
    } catch (err) {
      console.warn('Camera access denied or unavailable:', err);
    }
  }

  stopCamera() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try {
        this.mediaRecorder.stop();
      } catch (_) {}
    }
    clearInterval(this.timerInterval);
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.cameraVideo) {
      this.cameraVideo.srcObject = null;
    }
  }

  startMiscRecording() {
    if (!this.mediaStream) return;
    this.recordedChunks = [];

    const mimeTypes = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
    const chosenMime = mimeTypes.find((m) => MediaRecorder.isTypeSupported(m)) || '';

    this.mediaRecorder = new MediaRecorder(this.mediaStream, chosenMime ? { mimeType: chosenMime } : {});

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        this.recordedChunks.push(e.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      this.uploadMiscRecordedBlob();
    };

    this.mediaRecorder.start(250);
    this.recordingStartTime = Date.now();

    this.btnStartRecord.style.display = 'none';
    this.btnStopRecord.style.display = 'inline-flex';
    this.recIndicator.style.display = 'flex';

    this.timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
      const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const secs = String(elapsed % 60).padStart(2, '0');
      this.recTimer.textContent = `${mins}:${secs}`;
    }, 500);
  }

  stopMiscRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
    clearInterval(this.timerInterval);
    this.btnStartRecord.style.display = 'inline-flex';
    this.btnStopRecord.style.display = 'none';
    this.recIndicator.style.display = 'none';
    this.recTimer.textContent = '00:00';
  }

  async uploadMiscRecordedBlob() {
    if (this.recordedChunks.length === 0) return;
    const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
    const formData = new FormData();
    formData.append('video', blob, 'misc_mocap_recording.webm');
    formData.append('video_id', 'misc_mocap');

    try {
      this.btnStartRecord.disabled = true;
      this.btnStartRecord.textContent = 'Saving...';

      const res = await fetch('/api/recordings/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const newRec = await res.json();
      this.expandedVideos.add('misc_mocap');
      await this.fetchVideos();
      this.selectRecording(newRec, null);
    } catch (err) {
      console.error('Failed to upload misc recording:', err);
      alert('Error saving recorded video.');
    } finally {
      this.btnStartRecord.disabled = false;
      this.btnStartRecord.innerHTML = '<span>🔴 Start Recording</span>';
    }
  }

  async uploadMiscFile(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append('video', file, file.name);
    formData.append('video_id', 'misc_mocap');

    try {
      this.mocapTestDropzone.innerHTML = `<span class="spinner"></span> Uploading ${this.escapeHtml(file.name)}...`;
      const res = await fetch('/api/recordings/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const newRec = await res.json();
      this.expandedVideos.add('misc_mocap');
      await this.fetchVideos();
      this.selectRecording(newRec, null);
    } catch (err) {
      console.error('Misc file upload failed:', err);
      alert('Failed to upload test video file.');
    } finally {
      this.mocapTestDropzone.innerHTML = `
        <p><strong>Or click to upload a test video</strong> (or drag &amp; drop an MP4/WebM here) for standalone Motion Capture</p>
        <span style="font-size:0.8rem; color:var(--text-dim);">(Saved under "Test Motion Capture (Misc.)" in the sidebar)</span>
        <input type="file" id="mocap-test-file-input" accept="video/*" style="display: none;" />
      `;
      this.mocapTestFileInput = document.getElementById('mocap-test-file-input');
      if (this.mocapTestFileInput) {
        this.mocapTestFileInput.addEventListener('change', (e) => {
          if (e.target.files.length > 0) {
            this.uploadMiscFile(e.target.files[0]);
          }
        });
      }
    }
  }

  // -------------------------------------------------------------
  // Add / Delete Teacher Videos
  // -------------------------------------------------------------
  async handleAddYoutubeVideo() {
    const url = (this.youtubeUrlInput.value || '').trim();
    const name = (this.youtubeNameInput.value || '').trim();
    if (!url) {
      alert('Please enter a YouTube URL.');
      return;
    }

    try {
      this.btnLoadYoutube.disabled = true;
      this.btnLoadYoutubeText.innerHTML = `<span class="spinner"></span> Downloading YouTube Video...`;

      const res = await fetch('/api/videos/youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, name: name || null }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const newVideo = await res.json();
      this.youtubeUrlInput.value = '';
      this.youtubeNameInput.value = '';

      await this.fetchVideos();
      this.expandedVideos.add(newVideo.id);
      const found = this.videos.find((v) => v.id === newVideo.id) || newVideo;
      this.selectTeacherVideo(found);
    } catch (err) {
      console.error('Failed to add YouTube video:', err);
      alert(`Failed to load YouTube video: ${err.message}`);
    } finally {
      this.btnLoadYoutube.disabled = false;
      this.btnLoadYoutubeText.textContent = '📥 Load YouTube Video';
    }
  }

  async handleUploadTeacherVideo(file) {
    if (!file) return;
    const customName = (this.uploadVideoNameInput.value || '').trim();
    const formData = new FormData();
    formData.append('video', file, file.name);
    if (customName) {
      formData.append('name', customName);
    }

    try {
      this.uploadDropzone.innerHTML = `<span class="spinner"></span> Uploading & preparing ${this.escapeHtml(file.name)}...`;
      const res = await fetch('/api/videos/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const newVideo = await res.json();
      this.uploadVideoNameInput.value = '';

      await this.fetchVideos();
      this.expandedVideos.add(newVideo.id);
      const found = this.videos.find((v) => v.id === newVideo.id) || newVideo;
      this.selectTeacherVideo(found);
    } catch (err) {
      console.error('Failed to upload teacher video:', err);
      alert('Failed to upload video file.');
    } finally {
      this.uploadDropzone.innerHTML = `
        <p><strong>Click to choose a video file</strong> or drag and drop an MP4 here</p>
        <span style="font-size:0.8rem; color:var(--text-dim);">(MP4, WebM, MOV)</span>
        <input type="file" id="file-upload-input" accept="video/*" style="display: none;" />
      `;
      this.fileUploadInput = document.getElementById('file-upload-input');
      if (this.fileUploadInput) {
        this.fileUploadInput.addEventListener('change', (e) => {
          if (e.target.files.length > 0) {
            this.handleUploadTeacherVideo(e.target.files[0]);
          }
        });
      }
    }
  }

  async deleteTeacherVideo(video) {
    try {
      const res = await fetch(`/api/videos/${video.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      this.expandedVideos.delete(video.id);
      await this.fetchVideos();

      if (this.selectedVideo && this.selectedVideo.id === video.id) {
        if (this.videos.length > 0) {
          this.expandedVideos.add(this.videos[0].id);
          this.selectTeacherVideo(this.videos[0]);
        } else {
          this.selectedVideo = null;
          this.selectedRecording = null;
          this.switchMode('add-video');
        }
      }
    } catch (err) {
      console.error('Failed to delete teacher video:', err);
      alert('Failed to delete video.');
    }
  }

  // -------------------------------------------------------------
  // Selection & View Switching
  // -------------------------------------------------------------
  selectTeacherVideo(video) {
    this.stopTrainingSession(true);
    this.stopPolling();
    this.stopKeyposePolling();

    this.selectedVideo = video;
    this.selectedRecording = null;
    this.checkpointEditorOpen = false;
    this.renderSidebar();
    this.switchMode('teacher-video');
  }

  selectRecording(rec, parentVideo = null) {
    this.stopTrainingSession(true);
    this.stopPolling();
    this.stopKeyposePolling();

    if (parentVideo) {
      this.selectedVideo = parentVideo;
      this.expandedVideos.add(parentVideo.id);
    } else if (rec.video_id && rec.video_id !== 'misc_mocap') {
      this.selectedVideo = this.videos.find((v) => v.id === rec.video_id) || null;
      this.expandedVideos.add(rec.video_id);
    } else {
      this.selectedVideo = null;
      this.expandedVideos.add('misc_mocap');
    }

    this.selectedRecording = rec;
    this.renderSidebar();

    if (rec.is_processed) {
      this.switchMode('processed');
    } else {
      this.switchMode('unprocessed');
    }
  }

  switchMode(mode) {
    this.currentMode = mode;
    this.stopPolling();

    if (mode !== 'teacher-video') {
      this.stopTrainingSession(true);
      this.stopKeyposePolling();
      if (this.teacherVideo) {
        this.teacherVideo.pause();
      }
    }

    if (mode === 'test-mocap') {
      this.initCamera();
    } else {
      this.stopCamera();
    }

    this.addVideoView.style.display = mode === 'add-video' ? 'flex' : 'none';
    this.teacherVideoView.style.display = mode === 'teacher-video' ? 'flex' : 'none';
    this.testMocapView.style.display = mode === 'test-mocap' ? 'flex' : 'none';
    this.unprocessedView.style.display = mode === 'unprocessed' ? 'flex' : 'none';
    this.processedView.style.display = mode === 'processed' ? 'flex' : 'none';

    if (mode === 'teacher-video' && this.selectedVideo) {
      this.setupTeacherVideoView();
    } else if (mode === 'unprocessed' && this.selectedRecording) {
      this.setupUnprocessedView();
    } else if (mode === 'processed' && this.selectedRecording) {
      this.setupProcessedView();
    }
  }

  // -------------------------------------------------------------
  // Teacher Video View + Integrated YouTube Extension Controls
  // -------------------------------------------------------------
  setupTeacherVideoView() {
    const video = this.selectedVideo;
    if (!video) return;

    this.teacherVideoTitle.textContent = video.name;
    this.teacherVideoSubtitle.textContent = `${video.source_type === 'youtube' ? 'YouTube Video' : 'Uploaded MP4'} • Added ${video.created_at}`;

    const targetSrc = `/teacher_videos/${video.filename}`;
    if (!this.teacherVideo.src.endsWith(targetSrc)) {
      this.teacherVideo.src = targetSrc;
      this.teacherVideo.load();
    }

    this.updateTeacherVideoProcessedState(video);

    if (video.status === 'processing') {
      this.setKeyposeProcessingUi(true);
      this.startKeyposePolling(video.id);
    } else {
      this.setKeyposeProcessingUi(false);
    }
  }

  updateTeacherVideoProcessedState(video) {
    this.teacherTrainingCard.style.display = 'flex';

    if (video.is_processed) {
      this.teacherKeyposeBadge.className = 'badge badge-processed';
      this.teacherKeyposeBadge.textContent = `✓ ${video.keypose_count || 0} Keyposes Ready`;
      this.teacherProcessCard.style.display = 'none';

      this.btnStartTraining.disabled = false;
      this.btnStartTrainingText.textContent = '🔴 Record & Copy Video';
      this.trainingHeadline.textContent = 'Ready to Practice & Record';
      this.trainingSubtext.innerHTML =
        'Click <strong>Record &amp; Copy Video</strong>. The video will pause at each keypose and only play forward as you match the pose! Your clean webcam video (without overlays) is automatically saved under this video\'s recordings.';

      this.refreshCheckpointTimelineUI();
    } else {
      this.teacherKeyposeBadge.className = 'badge badge-unprocessed';
      this.teacherKeyposeBadge.textContent = '⏳ Needs Keypose Processing';
      this.teacherProcessCard.style.display = 'flex';

      // Strictly block recording until the teacher video is processed
      this.btnStartTraining.disabled = true;
      this.btnStartTrainingText.textContent = '🔒 Process Video First to Record';
      this.trainingHeadline.textContent = '🔒 Recording Locked Until Video is Processed';
      this.trainingSubtext.innerHTML =
        'Recordings for this video cannot be taken until you click <strong>Process Video (Extract Keyposes)</strong> above. (For standalone motion capture without a teacher video, click <strong>Test Motion Capture</strong> in the sidebar.)';

      this.segmentOverlayContainer.style.display = 'none';
      this.segmentBtnContainer.style.display = 'none';
      this.checkpointSavedPill.style.display = 'none';
    }
  }

  refreshCheckpointTimelineUI() {
    const video = this.selectedVideo;
    if (!video || !video.is_processed) {
      this.segmentOverlayContainer.style.display = 'none';
      this.segmentBtnContainer.style.display = 'none';
      this.checkpointSavedPill.style.display = 'none';
      return;
    }

    const duration = this.teacherVideo.duration;
    if (!duration || !Number.isFinite(duration) || duration <= 0) {
      return;
    }

    this.checkpointSlider.max = duration;

    const hasSavedCheckpoint = video.checkpoint_interval !== undefined && video.checkpoint_interval !== null;
    if (this.checkpointEditorOpen || !hasSavedCheckpoint) {
      const initialVal = hasSavedCheckpoint ? parseFloat(video.checkpoint_interval) : Math.min(duration, Math.max(2, duration / 4));
      this.checkpointSlider.value = initialVal;
      this.updateSegmentMarkers(initialVal);
      this.segmentOverlayContainer.style.display = 'block';
      this.segmentBtnContainer.style.display = 'flex';
      this.checkpointSavedPill.style.display = 'none';
    } else {
      this.updateSegmentMarkers(parseFloat(video.checkpoint_interval));
      this.segmentOverlayContainer.style.display = 'none';
      this.segmentBtnContainer.style.display = 'none';
      this.checkpointSavedPill.style.display = 'flex';
      this.checkpointSavedText.textContent =
        video.checkpoint_interval > 0
          ? `Checkpoint Interval: ${Number(video.checkpoint_interval).toFixed(2)}s (Every ${video.checkpoint_poses} poses)`
          : `Checkpoint: Full Video (${video.keypose_count} poses)`;
    }
  }

  updateSegmentMarkers(val) {
    const max = parseFloat(this.checkpointSlider.max || '0');
    this.checkpointGhostContainer.innerHTML = '';

    if (!max || max <= 0 || !val || val <= 0) {
      this.checkpointPrimaryBar.style.left = '0%';
      this.checkpointTimeLabel.textContent = 'Full Video';
      return;
    }

    const primaryPercent = Math.min(100, Math.max(0, (val / max) * 100));
    this.checkpointPrimaryBar.style.left = `${primaryPercent}%`;
    this.checkpointTimeLabel.textContent = `${val.toFixed(2)}s`;

    if (val >= 0.5) {
      let mult = 2;
      while (val * mult < max) {
        const ghostTime = val * mult;
        const ghostPercent = (ghostTime / max) * 100;

        const ghostBar = document.createElement('div');
        ghostBar.className = 'checkpoint-ghost-bar';
        ghostBar.style.left = `${ghostPercent}%`;

        const ghostTri = document.createElement('div');
        ghostTri.className = 'checkpoint-ghost-triangle';
        ghostBar.appendChild(ghostTri);

        this.checkpointGhostContainer.appendChild(ghostBar);
        mult += 1;
        if (mult > 50) break;
      }
    }
  }

  async saveVideoCheckpoint(intervalSec) {
    if (!this.selectedVideo) return;
    try {
      const res = await fetch(`/api/videos/${this.selectedVideo.id}/checkpoint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_interval: intervalSec }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      this.selectedVideo.checkpoint_interval = intervalSec !== null ? intervalSec : 0;
      this.selectedVideo.checkpoint_poses = data.checkpoint_poses;
      this.checkpointEditorOpen = false;
      this.refreshCheckpointTimelineUI();
    } catch (err) {
      console.error('Failed to save checkpoint:', err);
    }
  }

  // -------------------------------------------------------------
  // Step 1: Process Teacher Video into Keyposes
  // -------------------------------------------------------------
  async startTeacherKeyposeProcessing() {
    if (!this.selectedVideo) return;
    const videoId = this.selectedVideo.id;

    try {
      this.setKeyposeProcessingUi(true);
      this.teacherKeyposeLogs.style.display = 'block';
      this.teacherKeyposeLogs.textContent = '[init] Requesting keypose & trajectory extraction...\n';

      const res = await fetch(`/api/videos/${videoId}/process`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      this.startKeyposePolling(videoId);
    } catch (err) {
      console.error('Keypose processing failed:', err);
      this.setKeyposeProcessingUi(false);
      alert('Failed to start keypose processing.');
    }
  }

  setKeyposeProcessingUi(isProcessing) {
    this.btnProcessKeyposes.disabled = isProcessing;
    if (isProcessing) {
      this.btnProcessKeyposesText.innerHTML = `<span class="spinner"></span> Extracting Keyposes...`;
    } else {
      this.btnProcessKeyposesText.textContent = '⚡ Process Video (Extract Keyposes)';
    }
  }

  startKeyposePolling(videoId) {
    this.stopKeyposePolling();
    this.keyposePollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/videos/${videoId}/status`);
        const status = await res.json();

        if (status.logs && status.logs.length > 0) {
          this.teacherKeyposeLogs.style.display = 'block';
          this.teacherKeyposeLogs.textContent = status.logs.join('\n');
          this.teacherKeyposeLogs.scrollTop = this.teacherKeyposeLogs.scrollHeight;
        }

        if (status.status === 'ready' && status.is_processed) {
          this.stopKeyposePolling();
          this.setKeyposeProcessingUi(false);
          await this.fetchVideos();
          const updated = this.videos.find((v) => v.id === videoId);
          if (updated) {
            this.selectedVideo = updated;
            this.checkpointEditorOpen = true;
            this.updateTeacherVideoProcessedState(updated);
          }
        } else if (status.status === 'error') {
          this.stopKeyposePolling();
          this.setKeyposeProcessingUi(false);
          await this.fetchVideos();
        }
      } catch (err) {
        console.warn('Keypose poll error:', err);
      }
    }, 800);
  }

  stopKeyposePolling() {
    if (this.keyposePollInterval) {
      clearInterval(this.keyposePollInterval);
      this.keyposePollInterval = null;
    }
  }

  // -------------------------------------------------------------
  // Step 2: Interactive Copy-Training & Clean Recording Session
  // -------------------------------------------------------------
  toggleTrainingLayout(forceSideBySide) {
    this.trainingSideBySide =
      typeof forceSideBySide === 'boolean' ? forceSideBySide : !this.trainingSideBySide;

    if (this.teacherPlayerContainer) {
      this.teacherPlayerContainer.classList.toggle('side-by-side', this.trainingSideBySide);
    }

    if (this.webcamOverlayLabel) {
      this.webcamOverlayLabel.textContent = this.trainingSideBySide
        ? 'Live Webcam (Side by Side)'
        : 'Translucent Webcam Overlay';
    }

    if (this.webcamOpacityControl) {
      this.webcamOpacityControl.style.display = this.trainingSideBySide ? 'none' : 'inline-flex';
    }

    if (this.btnExpandOverlay) {
      this.btnExpandOverlay.style.display = this.trainingSideBySide ? 'none' : 'inline-block';
    }

    const toggleText = this.trainingSideBySide ? '🔀 Translucent Overlay' : '📐 Side by Side';
    if (this.btnToggleTrainingLayout) {
      this.btnToggleTrainingLayout.textContent = toggleText;
    }
    if (this.btnTrainingLayoutBar) {
      this.btnTrainingLayoutBar.innerHTML = `<span>${toggleText}</span>`;
    }
  }

  startTrainingSession() {
    if (!this.selectedVideo || !this.selectedVideo.is_processed) {
      alert('Please process the video into keyposes before taking a recording.');
      return;
    }
    this.stopTrainingSession(true);

    // Show Webcam Overlay (defaulting to full-video translucent overlay or side-by-side) & Telemetry Bar
    this.webcamOverlay.style.display = 'block';
    this.webcamOverlay.classList.add('expanded');
    if (this.btnExpandOverlay) {
      this.btnExpandOverlay.textContent = '🗗 Corner PIP';
    }
    if (this.overlayOpacitySlider) {
      this.webcamOverlay.style.setProperty('--webcam-overlay-opacity', this.overlayOpacitySlider.value || '0.45');
    }
    this.toggleTrainingLayout(this.trainingSideBySide);

    this.opencvOverlayImg.style.display = 'none';
    this.opencvOverlayStatus.style.display = 'flex';
    this.opencvOverlayStatus.textContent = 'Starting camera & pose detector...';

    this.btnStartTraining.style.display = 'none';
    if (this.btnTrainingLayoutBar) {
      this.btnTrainingLayoutBar.style.display = 'inline-flex';
    }
    this.btnSkipPose.style.display = 'inline-flex';
    this.btnStopTraining.style.display = 'inline-flex';
    this.trainingTelemetryBar.style.display = 'flex';
    this.trainingHeadline.textContent = '🔴 Recording & Copy-Training Active';
    this.trainingSubtext.textContent =
      'Match each keypose to make the video play forward! Toggle between Translucent Overlay and Side-by-Side anytime.';

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/training/${this.selectedVideo.id}`;
    this.trainingWs = new WebSocket(wsUrl);

    this.trainingWs.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleTrainingWsMessage(msg);
      } catch (err) {
        console.error('Invalid WS message:', err);
      }
    };

    this.trainingWs.onerror = (err) => {
      console.error('Training WebSocket error:', err);
    };

    this.trainingWs.onclose = () => {
      this.cancelPlayUntilRaf();
    };
  }

  handleTrainingWsMessage(msg) {
    if (msg.action === 'pause') {
      this.cancelPlayUntilRaf();
      this.teacherVideo.pause();
      return;
    }

    if (msg.action === 'seek') {
      this.cancelPlayUntilRaf();
      if (typeof msg.seek_time === 'number') {
        this.teacherVideo.currentTime = msg.seek_time;
      }
      return;
    }

    if (msg.action === 'play') {
      this.cancelPlayUntilRaf();
      this.teacherVideo.play().catch(() => {});
      return;
    }

    if (msg.action === 'play_until') {
      const targetTime = Number(msg.target_time);
      this.cancelPlayUntilRaf();
      this.teacherVideo.play().catch(() => {});

      const checkTime = () => {
        if (!this.trainingWs || this.trainingWs.readyState !== WebSocket.OPEN) return;
        if (this.teacherVideo.currentTime >= targetTime || this.teacherVideo.ended) {
          this.teacherVideo.pause();
          this.teacherVideo.currentTime = targetTime;
          this.playUntilRaf = null;
          this.trainingWs.send(JSON.stringify({ type: 'ARRIVED' }));
        } else {
          this.playUntilRaf = requestAnimationFrame(checkTime);
        }
      };
      this.playUntilRaf = requestAnimationFrame(checkTime);
      return;
    }

    if (msg.type === 'OPENCV_FRAME') {
      if (msg.image) {
        this.opencvOverlayImg.src = `data:image/jpeg;base64,${msg.image}`;
        this.opencvOverlayImg.style.display = 'block';
        this.opencvOverlayStatus.style.display = 'none';
      }

      const poseNum = typeof msg.pose_id === 'number' ? msg.pose_id + 1 : (msg.pose_index || 1);
      const errVal = typeof msg.score_ema === 'number' ? msg.score_ema : msg.error_score;
      const reqHold = msg.required_hold_frames || msg.hold_required || 8;

      this.telemetryPose.textContent = `Pose: ${poseNum} / ${msg.total_poses || '-'}`;
      this.telemetryState.textContent = `State: ${msg.state || 'PAUSED'}`;
      this.telemetryError.textContent =
        typeof errVal === 'number' ? `Error: ${errVal.toFixed(3)}` : 'Error: --';
      this.telemetryHold.textContent = `Hold: ${msg.hold_frames || 0} / ${reqHold}`;
      this.telemetryDtw.textContent =
        typeof msg.dtw_score === 'number' && msg.dtw_score > 0
          ? `Movement Score: ${msg.dtw_score}%`
          : 'Movement Score: --';

      const isCheckpoint = msg.state === 'CHECKPOINT';
      this.btnCheckpointContinue.style.display = isCheckpoint ? 'inline-flex' : 'none';
      this.btnCheckpointRetry.style.display = isCheckpoint ? 'inline-flex' : 'none';
      return;
    }

    if (msg.type === 'ERROR') {
      alert(msg.message || 'Training session error.');
      this.resetTrainingUi();
      return;
    }

    if (msg.type === 'SESSION_COMPLETE') {
      this.cancelPlayUntilRaf();
      this.teacherVideo.pause();
      this.resetTrainingUi();

      const savedRec = msg.recording;
      this.fetchVideos().then(() => {
        if (savedRec && savedRec.id) {
          this.expandedVideos.add(savedRec.video_id);
          const parentVid = this.videos.find((v) => v.id === savedRec.video_id);
          const matchingRec =
            parentVid && parentVid.recordings
              ? parentVid.recordings.find((r) => r.id === savedRec.id) || savedRec
              : savedRec;
          this.selectRecording(matchingRec, parentVid);
        }
      });
    }
  }

  sendSkipPose() {
    if (this.trainingWs && this.trainingWs.readyState === WebSocket.OPEN) {
      this.trainingWs.send(JSON.stringify({ type: 'SKIP_POSE' }));
    }
  }

  sendCheckpointAction(action) {
    if (this.trainingWs && this.trainingWs.readyState === WebSocket.OPEN) {
      this.trainingWs.send(JSON.stringify({ type: 'CHECKPOINT_ACTION', action }));
    }
  }

  stopTrainingSession(silent = false) {
    this.cancelPlayUntilRaf();
    if (this.trainingWs) {
      if (!silent && this.trainingWs.readyState === WebSocket.OPEN) {
        try {
          this.trainingWs.send(JSON.stringify({ type: 'STOP_SESSION' }));
          return;
        } catch (_) {}
      }
      try {
        this.trainingWs.close();
      } catch (_) {}
      this.trainingWs = null;
    }
    this.resetTrainingUi();
  }

  cancelPlayUntilRaf() {
    if (this.playUntilRaf) {
      cancelAnimationFrame(this.playUntilRaf);
      this.playUntilRaf = null;
    }
  }

  resetTrainingUi() {
    this.webcamOverlay.style.display = 'none';
    this.webcamOverlay.classList.add('expanded');
    if (this.teacherPlayerContainer) {
      this.teacherPlayerContainer.classList.remove('side-by-side');
    }
    if (this.btnExpandOverlay) {
      this.btnExpandOverlay.textContent = '🗗 Corner PIP';
    }
    this.btnStartTraining.style.display = 'inline-flex';
    if (this.btnTrainingLayoutBar) {
      this.btnTrainingLayoutBar.style.display = 'none';
    }
    this.btnSkipPose.style.display = 'none';
    this.btnCheckpointContinue.style.display = 'none';
    this.btnCheckpointRetry.style.display = 'none';
    this.btnStopTraining.style.display = 'none';
    this.trainingTelemetryBar.style.display = 'none';

    if (this.selectedVideo) {
      this.updateTeacherVideoProcessedState(this.selectedVideo);
    }
  }

  // -------------------------------------------------------------
  // Delete Recording (Renumbers 1..N within that Teacher Video or Misc)
  // -------------------------------------------------------------
  async deleteRecording(rec) {
    if (rec.status === 'processing') {
      alert('Cannot delete a recording while it is currently processing.');
      return;
    }

    try {
      const res = await fetch(`/api/recordings/${rec.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      await this.fetchVideos();

      if (this.selectedRecording && this.selectedRecording.id === rec.id) {
        if (rec.video_id === 'misc_mocap') {
          if (this.miscRecordings.length > 0) {
            this.selectRecording(this.miscRecordings[0], null);
          } else {
            this.switchMode('test-mocap');
          }
          return;
        }

        const parentVid = this.videos.find((v) => v.id === rec.video_id);
        if (parentVid && parentVid.recordings && parentVid.recordings.length > 0) {
          this.selectRecording(parentVid.recordings[0], parentVid);
        } else if (parentVid) {
          this.selectTeacherVideo(parentVid);
        } else if (this.videos.length > 0) {
          this.selectTeacherVideo(this.videos[0]);
        } else {
          this.switchMode('add-video');
        }
      } else if (this.selectedRecording) {
        let updated = null;
        if (this.selectedRecording.video_id === 'misc_mocap') {
          updated = this.miscRecordings.find((r) => r.id === this.selectedRecording.id);
        } else {
          const parentVid = this.videos.find((v) => v.id === this.selectedRecording.video_id);
          updated =
            parentVid && parentVid.recordings
              ? parentVid.recordings.find((r) => r.id === this.selectedRecording.id)
              : null;
        }
        if (updated) {
          this.selectedRecording = updated;
          if (this.currentMode === 'processed') {
            this.processedTitle.innerHTML = `<span>Recording #${updated.recording_no}</span> <span class="badge badge-processed">✓ Processed</span>`;
          } else if (this.currentMode === 'unprocessed') {
            this.unprocessedTitle.textContent = `Recording #${updated.recording_no}`;
          }
        }
      }
    } catch (err) {
      console.error('Failed to delete recording:', err);
      alert('Failed to delete recording.');
    }
  }

  // -------------------------------------------------------------
  // Unprocessed Recording View & Motion Capture Pipeline
  // -------------------------------------------------------------
  setupUnprocessedView() {
    const rec = this.selectedRecording;
    this.unprocessedTitle.textContent = `Recording #${rec.recording_no} (${rec.video_name || 'User Recording'})`;
    this.unprocessedDate.textContent = rec.created_at;
    this.unprocessedVideo.src = `/recordings/${rec.filename}`;

    if (this.downloadRawUserVideo) {
      this.downloadRawUserVideo.href = `/recordings/${rec.filename}`;
      this.downloadRawUserVideo.download = `recording_${rec.recording_no}_clean.mp4`;
    }

    this.consoleLogs.textContent = 'Ready to process clean user video into 3D Y Bot animation.';
    this.processStageText.textContent = '';

    if (rec.status === 'processing') {
      this.setProcessingUiState(true);
      this.startPolling(rec.id);
    } else {
      this.setProcessingUiState(false);
    }
  }

  async startProcessing() {
    if (!this.selectedRecording) return;
    const recId = this.selectedRecording.id;

    try {
      this.setProcessingUiState(true);
      this.consoleLogs.textContent = '[init] Requesting motion capture pipeline execution...\n';

      const res = await fetch(`/api/recordings/${recId}/process`, { method: 'POST' });
      await res.json();

      this.startPolling(recId);
    } catch (err) {
      console.error('Processing request failed:', err);
      this.setProcessingUiState(false);
      alert('Failed to start motion capture pipeline.');
    }
  }

  setProcessingUiState(isProcessing) {
    this.btnProcess.disabled = isProcessing;
    if (isProcessing) {
      this.btnProcessText.innerHTML = `<span class="spinner"></span> Processing Pipeline...`;
      this.processStageText.textContent = 'Running MediaPipe and Blender...';
    } else {
      this.btnProcessText.innerHTML = `🚀 Process Motion Capture`;
    }
  }

  startPolling(recId) {
    this.stopPolling();
    this.pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/recordings/${recId}/status`);
        const status = await res.json();

        if (status.stage) {
          this.processStageText.textContent = status.stage;
        }
        if (status.logs && status.logs.length > 0) {
          this.consoleLogs.textContent = status.logs.join('\n');
          this.consoleLogs.scrollTop = this.consoleLogs.scrollHeight;
        }

        if (status.status === 'completed' && status.is_processed) {
          this.stopPolling();
          await this.fetchVideos();
          const miscFound = this.miscRecordings.find((r) => r.id === recId);
          if (miscFound) {
            this.selectRecording(miscFound, null);
            return;
          }
          for (const v of this.videos) {
            const found = (v.recordings || []).find((r) => r.id === recId);
            if (found) {
              this.selectRecording(found, v);
              break;
            }
          }
        } else if (status.status === 'error') {
          this.stopPolling();
          this.setProcessingUiState(false);
          this.processStageText.textContent = 'Error occurred during processing.';
          await this.fetchVideos();
        }
      } catch (err) {
        console.warn('Status poll error:', err);
      }
    }, 800);
  }

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  // -------------------------------------------------------------
  // Processed Three.js 3D View (Keypose-Synced with Uploaded Video)
  // -------------------------------------------------------------
  setupProcessedView() {
    const rec = this.selectedRecording;
    this.processedTitle.innerHTML = `<span>Recording #${rec.recording_no}</span> <span class="badge badge-processed">✓ Processed</span>`;
    this.processedDate.textContent = `${rec.video_name || 'Video'} • ${rec.created_at}`;

    const parentVideo =
      rec && rec.video_id && rec.video_id !== 'misc_mocap'
        ? this.selectedVideo || this.videos.find((v) => v.id === rec.video_id) || null
        : null;
    this.isTeacherSyncedRecording = Boolean(
      rec && rec.video_id && rec.video_id !== 'misc_mocap'
    );
    const teacherFilename =
      (parentVideo && parentVideo.filename) || (rec && rec.teacher_video_filename) || null;
    const teacherDisplayName =
      (parentVideo && parentVideo.name) || (rec && rec.video_name) || 'Uploaded Video';

    if (this.downloadUserMp4) {
      this.downloadUserMp4.href = `/recordings/${rec.filename}`;
      this.downloadUserMp4.download = `recording_${rec.recording_no}_clean.mp4`;
    }

    if (rec.glb_filename) {
      this.downloadGlb.href = `/outputs/${rec.output_dir_name}/${rec.glb_filename}`;
      this.downloadGlb.style.display = 'inline-block';
    } else {
      this.downloadGlb.style.display = 'none';
    }

    if (rec.blend_filename) {
      this.downloadBlend.href = `/outputs/${rec.output_dir_name}/${rec.blend_filename}`;
      this.downloadBlend.style.display = 'inline-block';
    } else {
      this.downloadBlend.style.display = 'none';
    }

    if (rec.fbx_filename) {
      this.downloadFbx.href = `/outputs/${rec.output_dir_name}/${rec.fbx_filename}`;
      this.downloadFbx.style.display = 'inline-block';
    } else {
      this.downloadFbx.style.display = 'none';
    }

    if (this.pipVideo) {
      this.pipVideo.pause();
      this.pipSyncState = { activeIdx: -1, phase: 'idle', lastSeekTarget: -1 };
      if (this.isTeacherSyncedRecording) {
        const sideUrl = teacherFilename
          ? `/teacher_videos/${teacherFilename}`
          : `/api/recordings/${rec.id}/side_video`;
        this.pipVideo.src = sideUrl;
        this.pipVideo.load();
        this.pipVideo.loop = false;
        if (this.videoPanel) {
          this.videoPanel.style.display = 'flex';
        }
        if (this.btnTogglePip) {
          this.btnTogglePip.classList.add('active');
        }
        if (this.videoPanelLabel) {
          this.videoPanelLabel.textContent = `Uploaded Video: ${teacherDisplayName}`;
        }
        if (this.pipKeyposeStatus) {
          this.pipKeyposeStatus.style.display = 'block';
        }
        if (this.keyposeStepBadge) {
          this.keyposeStepBadge.style.display = 'inline-block';
        }
        if (this.btnPrevCheckpoint) {
          this.btnPrevCheckpoint.style.display = 'inline-flex';
        }
        if (this.btnNextCheckpoint) {
          this.btnNextCheckpoint.style.display = 'inline-flex';
        }
      } else {
        this.pipVideo.src = `/recordings/${rec.filename}`;
        this.pipVideo.load();
        this.pipVideo.loop = this.btnLoop.classList.contains('active');
        if (this.videoPanelLabel) {
          this.videoPanelLabel.textContent = 'User Recording (Clean)';
        }
        if (this.pipKeyposeStatus) {
          this.pipKeyposeStatus.style.display = 'none';
        }
        if (this.keyposeStepBadge) {
          this.keyposeStepBadge.style.display = 'none';
        }
        if (this.btnPrevCheckpoint) {
          this.btnPrevCheckpoint.style.display = 'none';
        }
        if (this.btnNextCheckpoint) {
          this.btnNextCheckpoint.style.display = 'none';
        }
      }
      this.pipVideo.playbackRate = parseFloat(this.speedSelect.value || '1.0');
    }

    this.initProcessedKeyposeSync(this.viewer && this.viewer.duration ? this.viewer.duration : 15.0);
    this.syncPipVideoWithMocap(0, false, true);

    if (!this.viewer) {
      this.viewer = new AvatarViewer('three-canvas-container', {
        onTimeUpdate: (current, duration) => {
          this.timelineScrubber.max = duration;
          this.timelineScrubber.value = current;
          this.timeLabel.textContent = `${this.formatTime(current)} / ${this.formatTime(duration)}`;
          this.syncPipVideoWithMocap(
            current,
            Boolean(this.viewer && this.viewer.isPlaying && !this.viewer.isSeeking),
            false
          );
        },
        onLoaded: (duration) => {
          this.timelineScrubber.max = duration;
          this.timelineScrubber.value = 0;
          this.timeLabel.textContent = `00:00 / ${this.formatTime(duration)}`;
          this.updatePlayButtonState(false);
          this.initProcessedKeyposeSync(duration);
          this.pipSyncState = { activeIdx: -1, phase: 'idle', lastSeekTarget: -1 };
          this.syncPipVideoWithMocap(0, false, true);
        },
        onFinished: () => {
          this.updatePlayButtonState(false);
          if (this.pipVideo) {
            this.pipVideo.pause();
          }
        },
      });
    } else {
      requestAnimationFrame(() => this.viewer.handleResize());
    }

    if (rec.glb_filename) {
      const glbUrl = `/outputs/${rec.output_dir_name}/${rec.glb_filename}`;
      this.viewer.loadModel(glbUrl);
    }
  }

  initProcessedKeyposeSync(mocapDuration) {
    this.processedKeyposeSync = [];
    if (this.mocapKeyposeDatalist) {
      this.mocapKeyposeDatalist.innerHTML = '';
    }

    const rec = this.selectedRecording;
    if (!this.isTeacherSyncedRecording || !rec) return;

    const parentVideo =
      this.selectedVideo || this.videos.find((v) => v.id === rec.video_id) || null;
    if (!parentVideo) return;

    const dur = Math.max(1.0, Number(mocapDuration) || 15.0);
    const rawSync = Array.isArray(rec.keypose_sync) ? rec.keypose_sync : [];
    const teacherEndFallback =
      this.pipVideo && Number.isFinite(this.pipVideo.duration) && this.pipVideo.duration > 0
        ? this.pipVideo.duration
        : null;

    if (rawSync.length > 0) {
      const lastNextArrive = Number(
        rawSync[rawSync.length - 1].next_mocap_arrive_time || dur
      );
      const scale = lastNextArrive > 0.1 ? dur / lastNextArrive : 1.0;

      this.processedKeyposeSync = rawSync.map((item, idx) => {
        const arrive = Number(item.mocap_arrive_time || 0) * scale;
        const nextArrive =
          idx + 1 === rawSync.length
            ? dur
            : Number(item.next_mocap_arrive_time || dur) * scale;
        const segDur = Math.max(0.25, nextArrive - arrive);
        // Brief checkpoint pause followed by smooth forward playback to the next checkpoint
        const pauseWindow =
          idx === 0
            ? Math.min(0.45, segDur * 0.2)
            : Math.min(0.75, segDur * 0.25);
        const match = Math.min(nextArrive - 0.1, arrive + pauseWindow);
        const tCur = Number(item.teacher_time || 0);
        let tNxt = Number(item.next_teacher_time ?? tCur);
        if (idx + 1 === rawSync.length && tNxt <= tCur) {
          tNxt =
            teacherEndFallback && teacherEndFallback > tCur
              ? teacherEndFallback
              : tCur + 1.0;
        }
        return {
          pose_id: Number(item.pose_id ?? idx),
          teacher_time: tCur,
          mocap_arrive_time: arrive,
          mocap_match_time: match,
          next_teacher_time: tNxt,
          next_mocap_arrive_time: nextArrive,
        };
      });
    } else if (parentVideo.timestamps && Object.keys(parentVideo.timestamps).length > 0) {
      // Fallback: build keypose segments proportionally from parentVideo.timestamps
      const sortedKeys = Object.keys(parentVideo.timestamps).sort(
        (a, b) => Number(a) - Number(b)
      );
      const total = sortedKeys.length;
      const weights = sortedKeys.map((k, idx) => {
        const tCur = Number(parentVideo.timestamps[k] || 0);
        const tNxt =
          idx + 1 < total
            ? Number(parentVideo.timestamps[sortedKeys[idx + 1]] || tCur)
            : tCur + 1.0;
        return 1.2 + Math.max(0, tNxt - tCur);
      });
      const totalWeight = weights.reduce((acc, w) => acc + w, 0) || total;

      let cursor = 0.0;
      this.processedKeyposeSync = sortedKeys.map((k, idx) => {
        const segDur = dur * (weights[idx] / totalWeight);
        const arrive = cursor;
        const nextArrive = idx + 1 === total ? dur : cursor + segDur;
        const tCur = Number(parentVideo.timestamps[k] || 0);
        let tNxt =
          idx + 1 < total
            ? Number(parentVideo.timestamps[sortedKeys[idx + 1]] || tCur)
            : tCur;
        if (idx + 1 === total && tNxt <= tCur) {
          tNxt =
            teacherEndFallback && teacherEndFallback > tCur
              ? teacherEndFallback
              : tCur + 1.0;
        }
        const pauseWindow =
          idx === 0
            ? Math.min(0.45, segDur * 0.2)
            : Math.min(0.75, segDur * 0.25);
        const match = Math.min(nextArrive - 0.1, arrive + pauseWindow);
        cursor = nextArrive;
        return {
          pose_id: Number(k),
          teacher_time: tCur,
          mocap_arrive_time: arrive,
          mocap_match_time: match,
          next_teacher_time: tNxt,
          next_mocap_arrive_time: nextArrive,
        };
      });
    }

    if (this.mocapKeyposeDatalist && this.processedKeyposeSync.length > 0) {
      this.processedKeyposeSync.forEach((seg, idx) => {
        const opt = document.createElement('option');
        opt.value = seg.mocap_arrive_time.toFixed(3);
        opt.label = `Keyframe ${idx + 1}`;
        this.mocapKeyposeDatalist.appendChild(opt);
      });
    }
  }

  getActiveCheckpointIndex(mocapTime) {
    if (!this.processedKeyposeSync || this.processedKeyposeSync.length === 0) {
      return 0;
    }
    const total = this.processedKeyposeSync.length;
    for (let i = 0; i < total; i += 1) {
      const seg = this.processedKeyposeSync[i];
      if (
        mocapTime >= seg.mocap_arrive_time - 0.001 &&
        (mocapTime < seg.next_mocap_arrive_time - 0.001 || i === total - 1)
      ) {
        return i;
      }
    }
    return 0;
  }

  stepProcessedCheckpoint(delta) {
    if (
      !this.isTeacherSyncedRecording ||
      !this.processedKeyposeSync ||
      this.processedKeyposeSync.length === 0
    ) {
      return;
    }
    const total = this.processedKeyposeSync.length;
    const currentMocapTime = parseFloat(this.timelineScrubber.value || '0');
    const curIdx =
      this.pipSyncState && this.pipSyncState.activeIdx >= 0
        ? this.pipSyncState.activeIdx
        : this.getActiveCheckpointIndex(currentMocapTime);
    const nextIdx = Math.max(0, Math.min(total - 1, curIdx + delta));
    const targetSeg = this.processedKeyposeSync[nextIdx];
    const targetMocapTime = targetSeg.mocap_arrive_time;

    this.timelineScrubber.value = targetMocapTime;
    if (this.viewer) {
      this.viewer.setKeyframeBarrier(targetSeg.next_mocap_arrive_time);
      this.viewer.seek(targetMocapTime);
    }
    this.pipSyncState = {
      activeIdx: nextIdx,
      phase: 'idle',
      lastSeekTarget: -1,
      bothReachedAt: 0,
    };
    this.syncPipVideoWithMocap(
      targetMocapTime,
      Boolean(this.viewer && this.viewer.isPlaying),
      true
    );
  }

  snapMocapTimeToKeypose(rawTime) {
    if (!this.processedKeyposeSync || this.processedKeyposeSync.length === 0) {
      return rawTime;
    }
    let bestEntry = this.processedKeyposeSync[0];
    let bestDist = Math.abs(rawTime - bestEntry.mocap_arrive_time);
    for (let i = 1; i < this.processedKeyposeSync.length; i += 1) {
      const candidate = this.processedKeyposeSync[i];
      const dist = Math.abs(rawTime - candidate.mocap_arrive_time);
      if (dist < bestDist) {
        bestDist = dist;
        bestEntry = candidate;
      }
    }
    return bestEntry.mocap_arrive_time;
  }

  seekPipVideoOnce(targetTime, force = false) {
    if (!this.pipVideo || this.pipVideo.readyState < 1) return;
    const clampedTarget = Math.max(
      0,
      Number.isFinite(this.pipVideo.duration) && this.pipVideo.duration > 0
        ? Math.min(this.pipVideo.duration - 0.01, targetTime)
        : targetTime
    );
    // Avoid aborting an in-flight asynchronous H.264 seek unless forced
    if (!force && this.pipVideo.seeking) return;
    if (
      !force &&
      Math.abs(this.pipSyncState.lastSeekTarget - clampedTarget) < 0.03 &&
      Math.abs(this.pipVideo.currentTime - clampedTarget) < 0.15
    ) {
      return;
    }
    try {
      this.pipSyncState.lastSeekTarget = clampedTarget;
      this.pipVideo.currentTime = clampedTarget;
    } catch (_) {}
  }

  syncPipVideoWithMocap(mocapTime, isMocapPlaying, forceSeek = false) {
    if (!this.pipVideo) return;

    // Standalone Misc recording (not tied to an uploaded teacher video)
    if (!this.isTeacherSyncedRecording || this.processedKeyposeSync.length === 0) {
      if (this.viewer) {
        this.viewer.setKeyframeBarrier(null);
      }
      if (
        this.videoPanel &&
        this.videoPanel.style.display !== 'none' &&
        this.pipVideo.readyState >= 1
      ) {
        if (
          !this.pipVideo.seeking &&
          Math.abs(this.pipVideo.currentTime - mocapTime) > 0.25
        ) {
          try {
            this.pipVideo.currentTime = mocapTime;
          } catch (_) {}
        }
        if (isMocapPlaying && this.pipVideo.paused) {
          this.pipVideo.play().catch(() => {});
        } else if (!isMocapPlaying && !this.pipVideo.paused) {
          this.pipVideo.pause();
        }
      }
      return;
    }

    const total = this.processedKeyposeSync.length;
    const viewerSpeed = parseFloat(this.speedSelect.value || '1.0');

    // If paused or scrubbing across keyframes:
    if (!isMocapPlaying) {
      const activeIdx = this.getActiveCheckpointIndex(mocapTime);
      const seg = this.processedKeyposeSync[activeIdx];
      if (this.viewer) {
        this.viewer.setKeyframeBarrier(seg.next_mocap_arrive_time);
      }
      if (this.keyposeStepBadge) {
        this.keyposeStepBadge.textContent = `Keyframe ${activeIdx + 1} / ${total}`;
      }
      if (!this.pipVideo.paused) {
        this.pipVideo.pause();
      }
      if (this.pipKeyposeStatus) {
        this.pipKeyposeStatus.textContent = `⏸ Keyframe #${activeIdx + 1}/${total} (${seg.teacher_time.toFixed(1)}s) — Paused at Keyframe`;
      }
      if (
        forceSeek ||
        this.pipSyncState.activeIdx !== activeIdx ||
        this.pipSyncState.phase !== 'PAUSED'
      ) {
        this.pipSyncState.activeIdx = activeIdx;
        this.pipSyncState.phase = 'PAUSED';
        this.pipSyncState.bothReachedAt = 0;
        this.seekPipVideoOnce(seg.teacher_time, true);
      }
      return;
    }

    // Active playback: initialize or re-anchor keyframe segment when starting playback or after a seek
    if (
      forceSeek ||
      this.pipSyncState.phase !== 'PLAYING' ||
      this.pipSyncState.activeIdx < 0 ||
      this.pipSyncState.activeIdx >= total
    ) {
      const startIdx =
        this.pipSyncState.activeIdx >= 0 &&
        this.pipSyncState.activeIdx < total &&
        Math.abs(
          mocapTime -
            this.processedKeyposeSync[this.pipSyncState.activeIdx].mocap_arrive_time
        ) < 0.08
          ? this.pipSyncState.activeIdx
          : this.getActiveCheckpointIndex(mocapTime);
      const startSeg = this.processedKeyposeSync[startIdx];

      this.pipSyncState.activeIdx = startIdx;
      this.pipSyncState.phase = 'PLAYING';
      this.pipSyncState.bothReachedAt = 0;

      if (this.viewer) {
        this.viewer.setKeyframeBarrier(startSeg.next_mocap_arrive_time);
      }

      this.pipVideo.playbackRate = viewerSpeed;
      if (
        this.pipVideo.currentTime < startSeg.teacher_time - 0.1 ||
        this.pipVideo.currentTime > startSeg.next_teacher_time + 0.1 ||
        forceSeek
      ) {
        this.seekPipVideoOnce(startSeg.teacher_time, true);
      }

      if (
        startSeg.next_teacher_time > startSeg.teacher_time + 0.02 &&
        this.pipVideo.paused
      ) {
        this.pipVideo.play().catch(() => {});
      }
    }

    const activeIdx = this.pipSyncState.activeIdx;
    const seg = this.processedKeyposeSync[activeIdx];
    const nextKeyframeNum = Math.min(total, activeIdx + 2);

    if (this.viewer) {
      this.viewer.setKeyframeBarrier(seg.next_mocap_arrive_time);
    }
    if (this.keyposeStepBadge) {
      this.keyposeStepBadge.textContent = `Keyframe ${activeIdx + 1} / ${total}`;
    }

    // Check whether the 3D mocap and the uploaded video have reached the target keypoint
    const mocapReached = mocapTime >= seg.next_mocap_arrive_time - 0.015;
    const teacherSpan = Math.max(0.0, seg.next_teacher_time - seg.teacher_time);
    const videoUnavailable = Boolean(
      this.pipVideo.error || this.pipVideo.readyState < 1
    );
    const teacherReached =
      videoUnavailable ||
      teacherSpan <= 0.02 ||
      this.pipVideo.currentTime >= seg.next_teacher_time - 0.025 ||
      this.pipVideo.ended;

    // If the uploaded video reached the next keyframe first, pause it there so it waits for the 3D mocap
    if (teacherReached) {
      if (!this.pipVideo.paused) {
        this.pipVideo.pause();
      }
    } else {
      // Otherwise keep the uploaded video playing toward seg.next_teacher_time at viewerSpeed
      if (Math.abs(this.pipVideo.playbackRate - viewerSpeed) > 0.01) {
        this.pipVideo.playbackRate = viewerSpeed;
      }
      if (this.pipVideo.paused && !this.pipVideo.seeking) {
        this.pipVideo.play().catch(() => {});
      }
    }

    // Case 1: Uploaded Video reached the next keyframe first -> waiting for 3D Mocap
    if (teacherReached && !mocapReached) {
      this.pipSyncState.bothReachedAt = 0;
      if (this.pipKeyposeStatus) {
        this.pipKeyposeStatus.textContent = `⏸ Keyframe #${nextKeyframeNum}/${total} (${seg.next_teacher_time.toFixed(1)}s) — Video Waiting for 3D Mocap (${mocapTime.toFixed(1)}s / ${seg.next_mocap_arrive_time.toFixed(1)}s)`;
      }
      return;
    }

    // Case 2: 3D Mocap reached the next keyframe first -> waiting for Uploaded Video
    if (mocapReached && !teacherReached) {
      this.pipSyncState.bothReachedAt = 0;
      if (this.pipKeyposeStatus) {
        this.pipKeyposeStatus.textContent = `⏸ Keyframe #${nextKeyframeNum}/${total} — 3D Mocap Waiting for Video (${this.pipVideo.currentTime.toFixed(1)}s / ${seg.next_teacher_time.toFixed(1)}s)`;
      }
      return;
    }

    // Case 3: Neither has reached the next keyframe yet -> both advancing toward Keyframe #nextKeyframeNum
    if (!mocapReached && !teacherReached) {
      this.pipSyncState.bothReachedAt = 0;
      if (this.pipKeyposeStatus) {
        this.pipKeyposeStatus.textContent = `▶ Moving to Keyframe #${nextKeyframeNum}/${total} (Video ${this.pipVideo.currentTime.toFixed(1)}s/${seg.next_teacher_time.toFixed(1)}s • Mocap ${mocapTime.toFixed(1)}s/${seg.next_mocap_arrive_time.toFixed(1)}s)`;
      }
      return;
    }

    // Case 4: BOTH have reached Keyframe #nextKeyframeNum!
    if (!this.pipSyncState.bothReachedAt) {
      this.pipSyncState.bothReachedAt = performance.now();
    }
    if (this.pipKeyposeStatus) {
      this.pipKeyposeStatus.textContent = `✓ Keyframe #${nextKeyframeNum}/${total} Reached by Both! Continuing...`;
    }

    // Brief 200ms synchronization beat at the keypoint before both advance to the next keypoint
    if (performance.now() - this.pipSyncState.bothReachedAt < 200) {
      return;
    }

    if (activeIdx + 1 < total) {
      const nextIdx = activeIdx + 1;
      const nextSeg = this.processedKeyposeSync[nextIdx];
      this.pipSyncState.activeIdx = nextIdx;
      this.pipSyncState.bothReachedAt = 0;

      // Release 3D mocap barrier to the next keypoint
      if (this.viewer) {
        this.viewer.setKeyframeBarrier(nextSeg.next_mocap_arrive_time);
      }

      // Advance uploaded video to the next keypoint
      this.pipVideo.playbackRate = viewerSpeed;
      if (Math.abs(this.pipVideo.currentTime - nextSeg.teacher_time) > 0.2) {
        this.seekPipVideoOnce(nextSeg.teacher_time, true);
      }
      if (nextSeg.next_teacher_time > nextSeg.teacher_time + 0.02) {
        this.pipVideo.play().catch(() => {});
      }
    } else {
      // Both have completed the final keyframe segment
      const isLoop = this.btnLoop && this.btnLoop.classList.contains('active');
      if (isLoop) {
        const firstSeg = this.processedKeyposeSync[0];
        this.pipSyncState.activeIdx = 0;
        this.pipSyncState.bothReachedAt = 0;
        if (this.viewer) {
          this.viewer.setKeyframeBarrier(firstSeg.next_mocap_arrive_time);
          this.viewer.seek(firstSeg.mocap_arrive_time);
        }
        this.seekPipVideoOnce(firstSeg.teacher_time, true);
        this.pipVideo.playbackRate = viewerSpeed;
        if (firstSeg.next_teacher_time > firstSeg.teacher_time + 0.02) {
          this.pipVideo.play().catch(() => {});
        }
      } else {
        if (this.viewer) {
          this.viewer.setKeyframeBarrier(null);
          this.viewer.pause();
        }
        if (!this.pipVideo.paused) {
          this.pipVideo.pause();
        }
        this.updatePlayButtonState(false);
        if (this.pipKeyposeStatus) {
          this.pipKeyposeStatus.textContent = `✓ All ${total} Keyframes Completed`;
        }
      }
    }
  }

  togglePlayback() {
    if (!this.viewer) return;
    const isPlaying = this.viewer.togglePlay();
    this.updatePlayButtonState(isPlaying);

    if (this.pipVideo) {
      const currentMocapTime = this.viewer.activeAction
        ? this.viewer.activeAction.time % (this.viewer.duration || 1)
        : 0;
      this.pipSyncState = { activeIdx: -1, phase: 'idle', lastSeekTarget: -1 };
      if (isPlaying) {
        this.syncPipVideoWithMocap(currentMocapTime, true, true);
      } else {
        this.pipVideo.pause();
        this.syncPipVideoWithMocap(currentMocapTime, false, true);
      }
    }
  }

  updatePlayButtonState(isPlaying) {
    if (isPlaying) {
      this.btnPlayIcon.textContent = '⏸';
      this.btnPlay.title = 'Pause Animation';
    } else {
      this.btnPlayIcon.textContent = '▶';
      this.btnPlay.title = 'Play Animation';
    }
  }

  formatTime(seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    const mins = String(Math.floor(s / 60)).padStart(2, '0');
    const secs = String(s % 60).padStart(2, '0');
    return `${mins}:${secs}`;
  }

  escapeHtml(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.mocapApp = new App();
});
