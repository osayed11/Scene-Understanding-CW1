"""Create a self-contained depth explorer and scientific figures from a cache.

No network inference is performed. Display clipping never changes stored maps.
The explorer contains sampled frames and a two-pixel grid of their raw metric
predictions; hover readouts name the exact sampled native pixel.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import ROOT
from part2.common import data_path


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KITTI depth explorer</title>
<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#17242c;background:#edf2f4}
body{max-width:1450px;margin:0 auto;padding:24px}h1{font-size:27px;margin:0 0 7px}p{line-height:1.5}
.muted{color:#53656d}.card{background:white;border:1px solid #d6e1e6;border-radius:12px;padding:18px;margin:18px 0}
.bar,.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.bar{justify-content:space-between}
.pill{font-size:13px;background:#e8f0fc;color:#163f77;border-radius:20px;padding:6px 11px}
.controls input[type=range]{flex:1;min-width:180px}button,select{font:inherit;padding:8px 12px;border:1px solid #b9cad3;border-radius:7px;background:white;cursor:pointer}
button:hover{background:#edf4fa}.images{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
canvas{display:block;width:100%;height:auto;background:#d9dde0;border-radius:5px;cursor:crosshair}
h2{font-size:16px;margin:0 0 8px}.legend{width:100%;height:15px;cursor:default;margin-top:12px}
.ticks{display:flex;justify-content:space-between;font-size:12px;margin-top:5px}.readout{font-family:ui-monospace,monospace;font-size:14px;min-height:23px;margin-top:15px}
.stats{display:flex;gap:30px;flex-wrap:wrap}.stats strong{display:block;font-size:20px}.stats span{font-size:13px;color:#53656d}
@media(max-width:850px){.images{grid-template-columns:1fr}body{padding:12px}}
</style>
<body><div class="bar"><div><h1 id="heading"></h1><div class="muted" id="subtitle"></div></div><span class="pill">No supplied depth ground truth</span></div>
<p><a href="preview.gif">Sampled animation</a> · <a href="overview.png">Four-frame overview</a> · <a href="sequence_stats.png">Sequence statistics</a></p>
<div class="card stats" id="stats"></div>
<div class="card"><div class="controls"><button id="prev" aria-label="Previous sample">Previous</button><button id="play">Play samples</button><button id="next" aria-label="Next sample">Next</button>
<input id="frame" type="range" aria-label="Sampled frame"><select id="choice" aria-label="Frame index"></select>
<label>Colour range <select id="range"><option value="20">0–20 m</option><option value="40">0–40 m</option><option value="80" selected>0–80 m</option><option value="160">0–160 m</option></select></label></div>
<p class="muted" id="frameLabel"></p>
<div class="images"><div><h2>RGB image</h2><canvas id="rgb"></canvas></div><div><h2>Predicted camera-z depth, metres</h2><canvas id="depth"></canvas>
<canvas id="legend" class="legend" width="256" height="12"></canvas><div class="ticks"><span>0 m · nearer</span><span id="middle"></span><span id="maximum"></span></div></div></div>
<div class="readout" id="readout">Move the pointer over either image to inspect predicted depth.</div>
<p class="muted" id="frameStats"></p></div>
<div class="card"><h2>How to interpret this</h2>
<p>Purple is nearer; green and yellow are farther away. Values above the selected colour range share its brightest colour. This clips the display only. Sky is assigned a finite background depth by the model; its colour is not a measured distance to the sky.</p>
<p>Change frames to inspect how the road, vehicles, buildings and thin structures are represented. The network processes a smaller image, so some edges and poles become smoother when its output is enlarged. Missing or nonpositive values would appear grey.</p>
<p>This is a sparse sample of the sequence. Adjacent slider positions are not necessarily adjacent video frames; use the displayed frame index and timestamp. Hover values are read from the original predictions on a two-pixel native grid. These images establish neither metric accuracy nor SLAM performance.</p>
</div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const palette=D.palette, slider=$('frame'), choice=$('choice');let index=0,timer=null;
$('heading').textContent=D.sequence+' · DA3 depth explorer';
$('subtitle').textContent=D.count+' completed frames · '+D.frames.length+' inspection samples · native '+D.native_width+' × '+D.native_height+' · network '+D.processed_width+' × '+D.processed_height;
const cards=[[D.count,'cached frames'],[D.valid_percent.toFixed(2)+'%','valid predicted pixels, from manifest'],[D.overflow_percent.toFixed(3)+'%','PNG overflow pixels, from manifest'],[D.depth_min.toFixed(2)+'–'+D.depth_max.toFixed(2)+' m','predicted extrema, including background']];
for(const [value,label] of cards){let block=document.createElement('div'),strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=label;block.append(strong,span);$('stats').append(block);}
slider.max=D.frames.length-1;slider.value=0;
D.frames.forEach((f,i)=>{let o=document.createElement('option');o.value=i;o.textContent='Frame '+f.index+(f.note?' · '+f.note:'');choice.append(o);});
function values(f){if(!f.values){let b=Uint8Array.from(atob(f.depth),c=>c.charCodeAt(0));f.values=new Float32Array(b.buffer);}return f.values;}
function render(){const f=D.frames[index], cap=Number($('range').value), a=values(f),c=$('depth');c.width=f.width;c.height=f.height;
let ctx=c.getContext('2d'),im=ctx.createImageData(f.width,f.height);
for(let i=0;i<a.length;i++){let good=Number.isFinite(a[i])&&a[i]>0,j=good?Math.round(Math.max(0,Math.min(1,a[i]/cap))*255):0;let color=good?palette[j]:[210,215,219];for(let k=0;k<3;k++)im.data[i*4+k]=color[k];im.data[i*4+3]=255;}
ctx.putImageData(im,0,0);$('rgb').width=f.width;$('rgb').height=f.height;
let image=new Image(),selected=index;image.onload=()=>{if(index===selected)$('rgb').getContext('2d').drawImage(image,0,0,f.width,f.height);};image.src=f.rgb;
$('frameLabel').textContent='Frame '+f.index+' / '+(D.count-1)+' · timestamp '+f.timestamp.toFixed(3)+' s'+(f.note?' · '+f.note:'');
$('frameStats').textContent='Predicted min '+f.minimum.toFixed(2)+' m · median '+f.median.toFixed(2)+' m · 95th percentile '+f.p95.toFixed(2)+' m · max '+f.maximum.toFixed(2)+' m';
$('middle').textContent=(cap/2)+' m';$('maximum').textContent=cap+'+ m · farther';slider.value=index;choice.value=index;
const l=$('legend').getContext('2d');for(let j=0;j<256;j++){l.fillStyle='rgb('+palette[j].join(',')+')';l.fillRect(j,0,1,12);}
}
function select(i){index=(i+D.frames.length)%D.frames.length;render();}
slider.oninput=()=>select(Number(slider.value));choice.onchange=()=>select(Number(choice.value));$('range').onchange=render;
$('prev').onclick=()=>select(index-1);$('next').onclick=()=>select(index+1);
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play samples';}else{timer=setInterval(()=>select(index+1),1100);$('play').textContent='Pause';}};
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){e.preventDefault();select(index-1);}if(e.key==='ArrowRight'){e.preventDefault();select(index+1);}});
for(const id of ['rgb','depth'])$(id).onpointermove=e=>{const f=D.frames[index],r=e.currentTarget.getBoundingClientRect(),x=Math.min(f.width-1,Math.max(0,Math.floor((e.clientX-r.left)/r.width*f.width))),y=Math.min(f.height-1,Math.max(0,Math.floor((e.clientY-r.top)/r.height*f.height))),z=values(f)[y*f.width+x];$('readout').textContent='Native pixel ('+(x*D.grid_step)+', '+(y*D.grid_step)+') · predicted depth '+(Number.isFinite(z)&&z>0?z.toFixed(3)+' m':'unavailable');};
render();
</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("kitti06",), default="kitti06")
    parser.add_argument("--pred-root", type=Path, default=ROOT / "depth_maps/da3_metric_large_cpu_float32")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--depth-max", type=float, default=80.)
    args = parser.parse_args(argv)
    if not np.isfinite(args.depth_max) or args.depth_max <= 0:
        parser.error("--depth-max must be finite and positive")
    cv2.setNumThreads(1)
    if hasattr(os, "nice"):
        os.nice(10)
    folder = args.pred_root / args.sequence
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest.get("full_complete"):
        raise ValueError("Use a complete sequence cache for the full-sequence explorer")
    rows = sorted(manifest["frames"].items(), key=lambda item: item[1]["index"])
    count = len(rows)
    if [r[1]["index"] for r in rows] != list(range(count)):
        raise ValueError("Expected complete original frame indices")
    sampled = set(np.rint(np.linspace(0, count-1, 12)).astype(int).tolist())
    notes = {}
    if args.sequence == "kitti06":
        notes = {46: "temporal pair A", 47: "temporal pair A", 308: "low frame maximum",
                 412: "temporal pair B", 413: "temporal pair B", 451: "high frame maximum",
                 1027: "near foreground"}
        sampled.update(i for i in notes if i < count)
        overview_indices = [0, 308, 451, 1027]
    else:
        overview_indices = np.rint(np.linspace(0, count-1, 4)).astype(int).tolist()
    sampled.update(overview_indices)
    samples, native = [], {}
    signature = manifest["signature"]
    lut = (plt.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8)
    for index in sorted(sampled):
        stem, row = rows[index]
        bgr = cv2.imread(str(data_path(row["rgb"], args.data_root)))
        depth = np.load(folder / (stem + ".npy"), allow_pickle=False)
        if bgr is None or bgr.shape[:2] != depth.shape:
            raise ValueError(f"Mismatched RGB/prediction: {stem}")
        small = np.ascontiguousarray(depth[::2, ::2], dtype="<f4")
        rgb = cv2.resize(bgr, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise OSError("Could not encode RGB preview")
        values = depth[np.isfinite(depth) & (depth > 0)]
        samples.append({"index": index, "timestamp": row["timestamp"], "note": notes.get(index, ""),
            "width": small.shape[1], "height": small.shape[0],
            "depth": base64.b64encode(small.tobytes()).decode(),
            "rgb": "data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95))})
        if index in overview_indices:
            native[index] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth)
    output = args.output_root or ROOT / "results/figures/part2" / (args.sequence + "_da3")
    output.mkdir(parents=True, exist_ok=True)
    camera = signature["camera"]
    processed = rows[0][1].get("preprocessing", {}).get("processed_shape", [None, None])
    payload = {"sequence": args.sequence, "count": count, "grid_step": 2,
        "native_width": camera["width"], "native_height": camera["height"],
        "processed_height": processed[0], "processed_width": processed[1],
        "valid_percent": 100 * (1-np.mean([r[1]["invalid_fraction"] for r in rows])),
        "overflow_percent": 100 * np.mean([r[1]["png_overflow_fraction"] for r in rows]),
        "depth_min": min(r[1]["depth_min_m"] for r in rows),
        "depth_max": max(r[1]["depth_max_m"] for r in rows), "palette": lut.tolist(), "frames": samples}
    (output / "viewer.html").write_text(HTML.replace("__PAYLOAD__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c")), encoding="utf-8")
    fig, axes = plt.subplots(len(overview_indices), 2, figsize=(14, 9.5), layout="constrained")
    for pair, index in zip(axes, overview_indices):
        rgb, depth = native[index]
        pair[0].imshow(rgb)
        pair[1].imshow(np.ma.masked_invalid(depth), cmap="viridis", vmin=0, vmax=args.depth_max)
        pair[0].set_title(f"RGB frame {index} · {rows[index][1]['timestamp']:.2f} s", loc="left", fontsize=10)
        pair[1].set_title("DA3 metric prediction" + (" · " + notes[index] if index in notes else ""), loc="left", fontsize=10)
        for ax in pair:
            ax.axis("off")
    mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, args.depth_max), cmap="viridis")
    fig.colorbar(mappable, ax=axes, label="Predicted camera-z depth (m); values above range display clipped", shrink=.85, extend="max", fraction=.025)
    fig.suptitle(f"{args.sequence}: RGB and DA3 depth on a shared 0–{args.depth_max:g} m scale", fontsize=15)
    fig.supxlabel("No supplied depth GT. Sky depth is a model background assignment. Stored predictions are unchanged.", fontsize=9)
    fig.savefig(output / "overview.png", dpi=150)
    plt.close(fig)
    indices = [r[1]["index"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, layout="constrained")
    axes[0].plot(indices, [r[1]["depth_max_m"] for r in rows], label="Maximum, including background", lw=1)
    axes[0].plot(indices, [r[1]["depth_min_m"] for r in rows], label="Minimum", lw=1)
    axes[0].set_ylabel("Predicted depth (m)")
    axes[0].legend(loc="upper right")
    axes[1].plot(indices, [r[1]["inference_seconds"] for r in rows], lw=.8)
    axes[1].set_ylabel("Inference time (s)")
    axes[1].set_xlabel("Original frame index")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"{args.sequence}: cache diagnostics · ranges are predictions, not accuracy measurements")
    fig.savefig(output / "sequence_stats.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"output_root": str(output), "sampled_frames": len(samples),
                      "viewer_bytes": (output / "viewer.html").stat().st_size, "new_inference": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
