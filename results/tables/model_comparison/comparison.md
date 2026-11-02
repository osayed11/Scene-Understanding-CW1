# Depth model sample comparison

These spread samples are diagnostic evidence, not a conclusive full-dataset model ranking. Only complete cases with full prediction coverage qualify for Pareto comparisons; samples, calibration, GT hashes, metric range and hardware must match. No weighted score is used.

Inference time excludes setup, warmup, file reads, saving and GT scoring. Per-sequence quality is a mean over GT frames. Overall quality is a macro mean over GT sequences; overall time in the sample-mean column averages all successful timed frames, including KITTI. KITTI supplies no GT depth. Per-sequence Pareto comparisons use mean seconds/frame; overall Pareto comparisons use estimated full-sequence hours, weighted by actual dataset frame counts, rather than the equally sampled sequence mix.

Full-sequence hours extrapolate the sample's mean time to 2000 Replica, 2585 TUM and 1101 KITTI frames, excluding overhead. They are estimates, not measured full runs.

| Model | Device / precision | Sequence | State | Frames / requested | GT | Sample mean s/frame | Raw AbsRel | Scaled RMSE m | Coverage | Est. hours | Pareto |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| da3_metric_large | cpu / float32 | replica_room0 | complete | 3 / 3 | 3 | 2.255 | 0.03319 | 0.09012 | 1 | 1.253 | True |
| da3_metric_large | cpu / float32 | tum_fr3 | complete | 3 / 3 | 3 | 3.06 | 0.09646 | 0.233 | 1 | 2.198 | True |
| da3_metric_large | cpu / float32 | kitti06 | complete | 3 / 3 | 0 | 1.361 | — | — | — | 0.4163 | not compared |
| da3_metric_large | cpu / float32 | overall | complete | 9 / 9 | 6 | 2.225 | 0.06483 | 0.1615 | 1 | 3.866 | True |
| da3_metric_large | cuda / float16 | replica_room0 | complete | 3 / 3 | 3 | 0.115 | 0.03329 | 0.09012 | 1 | 0.0639 | True |
| da3_metric_large | cuda / float16 | tum_fr3 | complete | 3 / 3 | 3 | 0.1651 | 0.09647 | 0.2328 | 1 | 0.1185 | True |
| da3_metric_large | cuda / float16 | kitti06 | complete | 3 / 3 | 0 | 0.05672 | — | — | — | 0.01735 | not compared |
| da3_metric_large | cuda / float16 | overall | complete | 9 / 9 | 6 | 0.1123 | 0.06488 | 0.1615 | 1 | 0.1998 | True |
| da3_metric_large | cuda / float32 | replica_room0 | complete | 3 / 3 | 3 | 0.1945 | 0.03319 | 0.09012 | 1 | 0.1081 | True |
| da3_metric_large | cuda / float32 | tum_fr3 | complete | 3 / 3 | 3 | 0.4878 | 0.09646 | 0.233 | 1 | 0.3503 | True |
| da3_metric_large | cuda / float32 | kitti06 | complete | 3 / 3 | 0 | 0.1188 | — | — | — | 0.03632 | not compared |
| da3_metric_large | cuda / float32 | overall | complete | 9 / 9 | 6 | 0.267 | 0.06483 | 0.1615 | 1 | 0.4947 | True |
| dav2_base | cpu / float32 | replica_room0 | complete | 3 / 3 | 3 | 4.085 | 0.3341 | 0.1844 | 1 | 2.269 | False |
| dav2_base | cpu / float32 | tum_fr3 | complete | 3 / 3 | 3 | 3.02 | 0.107 | 0.3038 | 1 | 2.168 | True |
| dav2_base | cpu / float32 | kitti06 | complete | 3 / 3 | 0 | 12.48 | — | — | — | 3.817 | not compared |
| dav2_base | cpu / float32 | overall | complete | 9 / 9 | 6 | 6.528 | 0.2206 | 0.2441 | 1 | 8.254 | False |
| dav2_large | cpu / float32 | replica_room0 | complete | 3 / 3 | 3 | 13.09 | 0.3272 | 0.1984 | 1 | 7.272 | False |
| dav2_large | cpu / float32 | tum_fr3 | complete | 3 / 3 | 3 | 9.329 | 0.1085 | 0.232 | 1 | 6.698 | False |
| dav2_large | cpu / float32 | kitti06 | complete | 3 / 3 | 0 | 36.01 | — | — | — | 11.01 | not compared |
| dav2_large | cpu / float32 | overall | complete | 9 / 9 | 6 | 19.48 | 0.2179 | 0.2152 | 1 | 24.98 | False |
| dav2_small | cpu / float32 | replica_room0 | complete | 3 / 3 | 3 | 1.573 | 0.3016 | 0.2307 | 1 | 0.8741 | True |
| dav2_small | cpu / float32 | tum_fr3 | complete | 3 / 3 | 3 | 1.036 | 0.1487 | 0.3751 | 1 | 0.7436 | True |
| dav2_small | cpu / float32 | kitti06 | complete | 3 / 3 | 0 | 4.363 | — | — | — | 1.334 | not compared |
| dav2_small | cpu / float32 | overall | complete | 9 / 9 | 6 | 2.324 | 0.2251 | 0.3029 | 1 | 2.952 | True |
| depth_pro | cpu / float32 | replica_room0 | complete | 3 / 3 | 3 | 72.43 | 0.02956 | 0.05652 | 1 | 40.24 | True |
| depth_pro | cpu / float32 | tum_fr3 | complete | 3 / 3 | 3 | 73.39 | 0.1185 | 0.3569 | 1 | 52.7 | False |
| depth_pro | cpu / float32 | kitti06 | complete | 3 / 3 | 0 | 71.67 | — | — | — | 21.92 | not compared |
| depth_pro | cpu / float32 | overall | complete | 9 / 9 | 6 | 72.5 | 0.07401 | 0.2067 | 1 | 114.9 | False |

GPU comparisons require fresh runs on these exact spread samples. The older three-adjacent-frame GPU smoke-test bundle is not imported or treated as comparable evidence.

## Case provenance

- `20474031352502d8`: hardware `{"cpu":"x86_64","device":"cpu","gpu":null,"platform":"Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35","torch":"2.14.1+cpu","torch_num_interop_threads":6,"torch_num_threads":6}`; files spread_da3_metric_large_general_cpu_float32.json; flags: none.
- `5c652123595a52ea`: hardware `{"cpu":"x86_64","device":"cuda","gpu":"Tesla T4","platform":"Linux-6.6.122+-x86_64-with-glibc2.35","torch":"2.11.0+cu128","torch_num_interop_threads":1,"torch_num_threads":1}`; files spread_da3_metric_large_cuda_float16.json; flags: none.
- `2fcccbccb6ab3f7d`: hardware `{"cpu":"x86_64","device":"cuda","gpu":"Tesla T4","platform":"Linux-6.6.122+-x86_64-with-glibc2.35","torch":"2.11.0+cu128","torch_num_interop_threads":1,"torch_num_threads":1}`; files spread_da3_metric_large_cuda_float32.json; flags: none.
- `21a8b6540d443aa0`: hardware `{"cpu":"x86_64","device":"cpu","gpu":null,"platform":"Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35","torch":"2.14.1+cpu","torch_num_interop_threads":6,"torch_num_threads":6}`; files spread_dav2_base_hypersim_cpu_float32.json, spread_dav2_base_vkitti_cpu_float32.json; flags: none.
- `11d659a0872c0823`: hardware `{"cpu":"x86_64","device":"cpu","gpu":null,"platform":"Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35","torch":"2.14.1+cpu","torch_num_interop_threads":6,"torch_num_threads":6}`; files spread_dav2_large_hypersim_cpu_float32.json, spread_dav2_large_vkitti_cpu_float32.json; flags: none.
- `19f6778fbc870da8`: hardware `{"cpu":"x86_64","device":"cpu","gpu":null,"platform":"Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35","torch":"2.14.1+cpu","torch_num_interop_threads":6,"torch_num_threads":6}`; files spread_dav2_small_hypersim_cpu_float32.json, spread_dav2_small_vkitti_cpu_float32.json; flags: none.
- `79c53f7ffac1817e`: hardware `{"cpu":"x86_64","device":"cpu","gpu":null,"platform":"Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35","torch":"2.14.1+cpu","torch_num_interop_threads":6,"torch_num_threads":6}`; files spread_depth_pro_general_cpu_float32.json; flags: none.
