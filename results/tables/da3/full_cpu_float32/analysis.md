# DA3 full CPU run analysis

Generated 2026-11-02T18:57:39.552852+00:00.

All metrics use the saved native float32 metre predictions. No new network inference is performed.

## Depth accuracy

| Dataset | Comparison | Scored frames | AbsRel | RMSE (m) | delta < 1.25 |
|---|---|---:|---:|---:|---:|
| TUM fr3 | raw | 2488 | 7.91% | 0.344 | 95.06% |
| TUM fr3 | median_scaled | 2488 | 5.45% | 0.314 | 96.34% |
| Replica room0 | raw | 2000 | 3.96% | 0.140 | 99.75% |
| Replica room0 | median_scaled | 2000 | 2.19% | 0.090 | 99.76% |

Arithmetic mean of per-frame metrics; valid GT > 0.001 m, no maximum depth or crop.

A separate GT-fitted scale for every frame; diagnoses depth shape and is not an online SLAM calibration.

## Figures

See `results/figures/part2/da3_full_cpu_float32/depth_quality.png` and `depth_examples.png` (PDF copies alongside).

The examples are selected by raw error over all scored frames: the middle-ranked frame and the highest-error frame in each indoor sequence.

## Scale variation

- TUM fr3: per-frame GT-fitted multiplier median 1.0316; 5th-95th percentile 0.9485-1.1385. A multiplier below 1 reduces overestimated depths.
- Replica room0: per-frame GT-fitted multiplier median 1.0329; 5th-95th percentile 1.0041-1.0700. A multiplier below 1 reduces overestimated depths.

The median multiplier alone cannot describe frame-to-frame scale variation. Improvements after fitting GT are diagnostic and do not establish that a fixed online correction will transfer to other views.

## Integrity

Audit status: passed; 0 errors. Verified 5686 NPY/PNG pairs and their source RGB hashes.
Associations still use portable data/... tokens; rebase paths before direct pySLAM use.

## Recorded inference time

| Dataset | Mean seconds/frame | Median | p90 | Inference hours |
|---|---:|---:|---:|---:|
| KITTI 06 | 1.803 | 1.629 | 2.250 | 0.551 |
| TUM fr3 | 4.067 | 3.808 | 4.765 | 2.920 |
| Replica room0 | 2.982 | 2.690 | 3.980 | 1.657 |

Inference timing excludes cache writes, input I/O, validation and interruption time. Session-level timing limits are in cache_diagnostics.json.

## Temporal consistency

Every adjacent pair is recorded; images are sampled on an 8-pixel grid. Errors below are equal-pair means of within-pair statistics, conditional on retained correspondences.

| Dataset | Primary pairs | Mean pair-median relative error | Mean pair-median absolute error (m) | All-pair retained/sample |
|---|---:|---:|---:|---:|
| KITTI 06 | 1100 | 2.70% | 0.5530 | 66.13% |
| TUM fr3 | 2389 | 1.17% | 0.0175 | 79.81% |
| Replica room0 | 1999 | 0.61% | 0.0170 | 98.80% |

Primary TUM/Replica errors use source GT geometry and target GT visibility. KITTI uses predictions and approximate reference poses, so its figure has different limitations. TUM fallback modes and missing poses remain explicit in the CSV and temporal_summary.json. Coverage above includes all available modes; missing-pose pairs have no sampled points.

## Sampled spatial patterns

SAMPLED spatial diagnostic:32 evenly spread GT-matched frames per indoor sequence;64frames total, not full-sequence spatial statistics

| Dataset | AbsRel in GT discontinuity band | AbsRel outside detected band | Band fraction of valid pixels |
|---|---:|---:|---:|
| TUM fr3 | 14.36% | 7.59% | 6.93% |
| Replica room0 | 7.57% | 3.81% | 3.66% |

On these 64 frames, relative error is approximately twice as large in the detected boundary bands. Edges use valid four-neighbor GT jumps >0.1 m +5% of nearer depth, dilated by 3 pixels. These are pooled sampled-pixel metrics, not the full-sequence frame means above; outside the band is not assumed planar. See spatial_diagnostics.json for exact frame indices, bin populations and signed errors, and spatial_errors.png for the plot.

## ORB diagnostic keypoints

These are fresh ORB detections, not the saved SLAM observations. Metrics pool all valid GT keypoint pairs and therefore use a different aggregation from dense metrics.

| Dataset | Keypoints | Keypoints with GT | AbsRel | 1x heuristic-sigma coverage | Sigma/absolute-error correlation |
|---|---:|---:|---:|---:|---:|
| KITTI 06 | 2201712 | 0 | unavailable | unavailable | unavailable |
| TUM fr3 | 4960839 | 4468466 | 8.77% | 18.08% | 0.403 |
| Replica room0 | 3953368 | 3951853 | 6.09% | 32.68% | 0.495 |

Patch dispersion is a heuristic, not a calibrated confidence interval. Map scale is unavailable because no paired triangulated map observations were supplied.

## Interpretation limits

- KITTI 06 has no supplied GT depth, so no KITTI depth accuracy is claimed.
- TUM uses nearest GT depth within 20 ms; unmatched RGB frames are excluded from depth scores and sensor holes are masked.
- KITTI temporal geometry/visibility is prediction-dependent and uses the supplied reference-pose stream without an explicit cam0-to-colour-cam2 extrinsic correction.
- Temporal consistency does not establish metric accuracy; moving objects and sensor timing can affect it.
- ORB keypoints are diagnostic detections, not the saved SLAM observations; patch dispersion is an uncalibrated uncertainty heuristic.
- Full inference and these diagnostics do not by themselves finish the actual SLAM-keypoint/map-scale requirements or Part 3 integrations.
