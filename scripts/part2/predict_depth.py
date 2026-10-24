"""Part 2: run ONE depth network (Depth Anything V2 / Depth Anything 3 / Depth Pro) on every frame of
every sequence and save 16-bit PNG depth in depth_maps/<sequence>/ (see scripts/depth.py for the convention).
Run inside the pySLAM depth environment:  cd third_party/pyslam && pixi run -e depth python ../../scripts/part2/predict_depth.py ...
Also writes a timestamps/association file so pySLAM can pair rgb and depth."""
