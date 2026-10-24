"""Part 2: depth + uncertainty at keypoints and a scale for the map.
Detect keypoints as pySLAM does, read depth there (depth.depth_at_keypoints), estimate an uncertainty,
and estimate the scale between predicted depth and the SLAM map / GT. Output: per-frame npz + table2_keypoints.csv"""
