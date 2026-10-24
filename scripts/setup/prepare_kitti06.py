"""Part 1: KITTI 06 needs no conversion. pySLAM reads the colour video with --video, the frame timestamps
(times.txt) and groundtruth.txt (t tx ty tz qx qy qz qw scale), all shipped with pySLAM in
data/videos/kitti06. scripts/measure_run.py reads the same groundtruth.txt (first 8 columns).
The frames are extracted from the video for the depth network by scripts/setup/download_data.sh."""
