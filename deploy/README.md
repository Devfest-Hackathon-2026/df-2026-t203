# Run MotionLearn

## Local development

Install Node.js 22, Python 3.11, FFmpeg, and Blender. Install the backend dependencies in a virtual environment with `python -m pip install -r deploy/requirements.txt`. The 3D pipeline also needs a Python environment with legacy MediaPipe (`mediapipe==0.10.9`, `numpy==1.23.5`, `opencv-contrib-python==4.11.0.86`) and one with `skellyforge @ git+https://github.com/freemocap/skellyforge@main`. Set `MEDIAPIPE_PYTHON` and `SKELLYFORGE_PYTHON` to those interpreters. Set `BLENDER_EXE` to the Blender executable if it is outside `PATH`. For thumbs-up and thumbs-down gestures, set `MOTIONLEARN_GESTURE_MODEL` to the [MediaPipe gesture model](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task).

Set `MOTIONLEARN_DATA_DIR` to a fresh directory such as `.motionlearn-data`, then run `python final-web-app/run_server.py` from the repository root. In `final-web-app`, run `npm ci` and `npm run dev`. Open the localhost URL printed by Vite. This data directory starts with the sample teacher video and an empty recording library. The checkpoint buttons also work when the gesture model is unavailable.

## Compute Engine

Use a billing-enabled Google Cloud project, a hostname you control, and Google Cloud CLI access with permission to create Compute Engine resources and connect through IAP. The commands below use Cloud Shell or Bash. [Linux VM setup](https://docs.cloud.google.com/compute/docs/create-linux-vm-instance) · [IAP SSH access](https://docs.cloud.google.com/iap/docs/using-tcp-forwarding)

```sh
PROJECT=your-project-id
REGION=asia-northeast1
ZONE=asia-northeast1-a
gcloud config set project "$PROJECT"
gcloud services enable compute.googleapis.com iap.googleapis.com
gcloud compute networks create motionlearn --subnet-mode=custom
gcloud compute networks subnets create motionlearn-tokyo --network=motionlearn \
  --region="$REGION" --range=10.42.0.0/24
gcloud compute firewall-rules create motionlearn-web --network=motionlearn \
  --allow=tcp:80,tcp:443 --source-ranges=0.0.0.0/0 --target-tags=motionlearn
gcloud compute firewall-rules create motionlearn-iap --network=motionlearn \
  --allow=tcp:22 --source-ranges=35.235.240.0/20 --target-tags=motionlearn
gcloud compute addresses create motionlearn-ip --region="$REGION"
IP=$(gcloud compute addresses describe motionlearn-ip --region="$REGION" --format='value(address)')
gcloud compute disks create motionlearn-data --zone="$ZONE" --size=50GB --type=pd-balanced
gcloud compute instances create motionlearn --zone="$ZONE" --machine-type=e2-standard-4 \
  --subnet=motionlearn-tokyo --address="$IP" --tags=motionlearn \
  --image-project=ubuntu-os-cloud --image-family=ubuntu-2404-lts-amd64 \
  --boot-disk-size=30GB --no-service-account --no-scopes \
  --disk=name=motionlearn-data,device-name=motionlearn-data,mode=rw,auto-delete=no
printf 'Point your hostname A record at %s\n' "$IP"
gcloud compute ssh motionlearn --zone="$ZONE" --tunnel-through-iap
```

On the VM, inspect and mount the disk. `mkfs` is a one-time step for the newly provisioned empty disk. [Mounting Persistent Disk](https://docs.cloud.google.com/compute/docs/disks/add-persistent-disk)

```sh
DISK=/dev/disk/by-id/google-motionlearn-data
sudo lsblk --fs "$DISK"
# On the new blank disk only:
sudo mkfs.ext4 "$DISK"
sudo mkdir -p /srv/motionlearn-data
UUID=$(sudo blkid -s UUID -o value "$DISK")
printf 'UUID=%s /srv/motionlearn-data ext4 defaults 0 2\n' "$UUID" | sudo tee -a /etc/fstab
sudo mount /srv/motionlearn-data
sudo install -d -o 10001 -g 10001 /srv/motionlearn-data/library
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
```

Transfer the source to the VM, including `final-web-app/public/media`. A source archive transferred with `gcloud compute scp --tunnel-through-iap` works for assets that are only in your working tree. Exclude build output, dependencies, credentials, and local recording data from the archive. In the deployed checkout, run the following commands from the repository root. The hash command prompts for the shared password. [Caddy authentication](https://caddyserver.com/docs/caddyfile/directives/basic_auth) · [Caddy TLS](https://caddyserver.com/docs/quick-starts/reverse-proxy)

```sh
sudo docker run --rm -it caddy:2.10 caddy hash-password
umask 077
touch deploy/.env
chmod 600 deploy/.env
# Edit deploy/.env with these settings. Keep the hash in single quotes.
# MOTIONLEARN_HOST=motionlearn.example.com
# MOTIONLEARN_PASSWORD_HASH='$2a$...'

sudo docker compose --env-file deploy/.env -f deploy/compose.yaml config --quiet
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml up --build -d
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml ps
```

Sign in as `judge` on the HTTPS site. Upload a short teacher clip, set a checkpoint interval shorter than the clip, and train from a separate laptop. Retry a checkpoint and check that the saved take and generated rig play in sync. Test the downloads and the unauthenticated response, then reboot the VM and check that the library survives. For a later release, rerun `up --build -d` against the same mounted data directory.
