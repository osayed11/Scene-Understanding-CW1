#!/usr/bin/env bash
# Fetch the three common sequences into data/ (~14 GB download; only Replica room0 is kept).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/data"

# KITTI 06: colour video, timestamps and ground truth as shipped with pySLAM
mkdir -p kitti06 && cd kitti06
for f in 06.txt groundtruth.txt times.txt video_color.mp4; do
  [ -f "$f" ] || curl -sL -o "$f" "https://raw.githubusercontent.com/luigifreda/pyslam/master/data/videos/kitti06/$f"
done
cd ..

# TUM fr3 long office household
mkdir -p tum && cd tum
if [ ! -d rgbd_dataset_freiburg3_long_office_household ]; then
  curl -L -o fr3.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
  tar xzf fr3.tgz && rm fr3.tgz
fi
cd ..

# Replica room0 (the zip holds all eight scenes, 12.4 GB; we keep room0 only)
mkdir -p replica && cd replica
if [ ! -d room0 ]; then
  curl -L -o Replica.zip https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
  unzip -q Replica.zip 'Replica/room0/*' && rm Replica.zip
  mv Replica/room0 . && rmdir Replica
fi

# KITTI 06 frames, extracted from the video for the depth network (Part 2)
cd "$ROOT/data/kitti06"
if [ ! -d image_2 ]; then
  mkdir -p image_2
  python3 -c "
import cv2
c=cv2.VideoCapture('video_color.mp4'); i=0
while True:
    ok,f=c.read()
    if not ok: break
    cv2.imwrite(f'image_2/{i:06d}.png', f); i+=1
print('extracted', i, 'frames')
"
fi
