"""Build tiny random-weight Ultralytics checkpoints offline (no downloads).

Run by test_lab_weights.py in a child process so the pytest process never imports torch:
    python build_checkpoints.py <out dir>
Writes seg.pt (classes renamed red/green/blue/wood/..., train imgsz 640), obb.pt, detect.pt,
cls.pt (an unsupported task), garbage.pt (random bytes) and plain.pt (a torch file that is
not a model).

A freshly initialised YOLO26 head predicts almost zero class probability everywhere, so
nothing would be detected and the conversion to cubes would never run. The class-branch
bias is therefore set so that class 0 ("red") fires on every anchor and the others never do:
the boxes, masks and rotated boxes are still random.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
os.environ.update({"YOLO_CONFIG_DIR": str(out / "cfg"), "YOLO_AUTOINSTALL": "False", "YOLO_OFFLINE": "1",
                   "YOLO_VERBOSE": "False"})

import torch  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402

settings.update({"sync": False})
torch.manual_seed(0)
os.chdir(out)  # anything Ultralytics writes relative to the cwd stays in the tmp dir

for yaml, name in [("yolo26n-seg.yaml", "seg.pt"), ("yolo26n-obb.yaml", "obb.pt"), ("yolo26n.yaml", "detect.pt"),
                   ("yolo26n-cls.yaml", "cls.pt")]:
    model = YOLO(yaml)
    model.model.names = {i: (["red", "green", "blue", "wood"][i] if i < 4 else f"c{i}") for i in range(len(model.names))}
    head = model.model.model[-1]
    for branch in ("cv3", "one2one_cv3"):
        for seq in getattr(head, branch, []):
            seq[-1].bias.data.fill_(-20.0)
            seq[-1].bias.data[0] = 20.0
    model.save(out / name)

ckpt = torch.load(out / "seg.pt", weights_only=False)
ckpt["train_args"] = {"imgsz": 640}
torch.save(ckpt, out / "seg.pt")
(out / "garbage.pt").write_bytes(os.urandom(4096))
torch.save({"hello": 1}, out / "plain.pt")
print("ok")
