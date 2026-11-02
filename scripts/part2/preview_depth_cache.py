"""Render a sampled RGB/depth preview without loading a network or changing its cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
try:
    os.nice(10)
except (AttributeError, OSError):
    pass

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT
from part2.common import load_frame_records

cv2.setNumThreads(1)


def render(rgb, depth, index, timestamp, *, stride=10, fps=5, maximum=80.0):
    width, row_height = 768, 232
    image = np.full((560, width, 3), 22, dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0)
    values = np.where(valid, np.clip(depth / maximum, 0, 1), 0)
    colors = cv2.applyColorMap(np.rint(values * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colors[~valid] = 112
    image[38:270] = cv2.resize(rgb, (width, row_height), interpolation=cv2.INTER_AREA)
    image[292:524] = cv2.resize(colors, (width, row_height), interpolation=cv2.INTER_NEAREST)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"KITTI 06 | frame {index:06d} | source t={timestamp:.3f} s", (12, 16), font, .48, (245,245,245), 1, cv2.LINE_AA)
    cv2.putText(image, f"SAMPLED PREVIEW | every {stride} frames | playback {fps:g} fps | RGB above", (12, 32), font, .40, (215,215,215), 1, cv2.LINE_AA)
    cv2.putText(image, f"DA3 metric depth | fixed 0 to {maximum:g} m | gray invalid | bright far: display clipped only", (12, 286), font, .39, (240,240,240), 1, cv2.LINE_AA)
    ramp = cv2.applyColorMap(np.linspace(0,255,720).astype(np.uint8)[None], cv2.COLORMAP_VIRIDIS)
    image[530:542,24:744] = np.repeat(ramp,12,axis=0)
    for fraction in (0,.25,.5,.75,1):
        text = f"{maximum*fraction:g} m"
        x = min(int(24+720*fraction), 718)
        cv2.putText(image,text,(x,555),font,.34,(230,230,230),1,cv2.LINE_AA)
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence",default="kitti06")
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--pred-root",type=Path,default=ROOT/"depth_maps/da3_metric_large_cpu_float32")
    parser.add_argument("--output",type=Path,default=ROOT/"results/figures/part2/kitti06_da3/preview.mp4")
    parser.add_argument("--stride",type=int,default=10)
    parser.add_argument("--fps",type=float,default=5)
    parser.add_argument("--display-max",type=float,default=80)
    args=parser.parse_args()
    if args.stride<1 or args.fps<=0 or args.display_max<=0:
        parser.error("stride, fps and display-max must be positive")
    frames=load_frame_records(args.sequence,data_root=args.data_root)
    selected=frames[::args.stride]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_name(args.output.stem+".tmp.mp4")
    ffmpeg=shutil.which("ffmpeg")
    encoder=None
    if ffmpeg:
        encoder=subprocess.Popen([ffmpeg,"-v","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s","768x560","-r",str(args.fps),"-i","-","-an","-c:v","libx264","-threads","1","-preset","medium","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(temporary)],stdin=subprocess.PIPE)
        writer=None
        codec="H.264/libx264"
    else:
        writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*"mp4v"),args.fps,(768,560))
        if not writer.isOpened():
            raise RuntimeError("No existing MP4 encoder available")
        codec="MPEG-4 Part 2/mp4v"
    gif_frames=[]
    sample_metadata=[]
    for frame in selected:
        rgb=cv2.imread(str(frame.rgb),cv2.IMREAD_COLOR)
        depth=np.load(args.pred_root/args.sequence/f"{frame.stem}.npy",allow_pickle=False,mmap_mode="r")
        if rgb is None or depth.shape!=rgb.shape[:2]:
            raise ValueError(f"RGB/depth mismatch for {frame.stem}")
        canvas=render(rgb,depth,frame.index,frame.timestamp,stride=args.stride,fps=args.fps,maximum=args.display_max)
        if encoder:
            encoder.stdin.write(canvas.tobytes())
        else:
            writer.write(canvas)
        small=cv2.resize(canvas,(512,374),interpolation=cv2.INTER_AREA)
        gif_frames.append(Image.fromarray(cv2.cvtColor(small,cv2.COLOR_BGR2RGB)).quantize(colors=128,method=Image.Quantize.FASTOCTREE))
        valid=np.isfinite(depth)&(depth>0)
        sample_metadata.append({"index":frame.index,"timestamp":frame.timestamp,"stem":frame.stem,
                                "valid_fraction":float(valid.mean()),
                                "fraction_above_display_cap":float(np.mean(valid&(depth>args.display_max)))})
    if encoder:
        encoder.stdin.close()
        if encoder.wait()!=0:
            raise RuntimeError("Existing ffmpeg failed to encode H.264")
    else:
        writer.release()
    temporary.replace(args.output)
    gif=args.output.with_suffix(".gif")
    gif_frames[0].save(gif,save_all=True,append_images=gif_frames[1:],duration=round(1000/args.fps),loop=0,optimize=False)
    duration=len(selected)/args.fps
    metadata={"sequence":args.sequence,"model":"da3_metric_large","cache":str(args.pred_root/args.sequence),
              "source":"cached NPY predictions; no network inference","sampling":"every Nth source frame, original identities retained",
              "stride":args.stride,"sample_count":len(selected),"fps":args.fps,"duration_seconds":duration,
              "source_span_seconds":selected[-1].timestamp-selected[0].timestamp,
              "approximate_time_compression":(selected[-1].timestamp-selected[0].timestamp)/duration,
              "dimensions":[768,560],"codec":codec,"video":args.output.name,"gif":gif.name,
              "browser_note":"Use GIF in browsers if MPEG-4 Part 2 playback is unavailable" if not ffmpeg else "H.264 video",
              "display_depth_range_metres":[0,args.display_max],"display_clipping_only":True,
              "invalid_color":"gray","first_sample":sample_metadata[0],"last_sample":sample_metadata[-1],
              "samples":sample_metadata}
    args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"gif":str(gif),"frames":len(selected),"duration":duration,"codec":codec},indent=2))


if __name__=="__main__":
    main()
