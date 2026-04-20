import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm
import cv2
import argparse

# ── Argumente ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",    type=str, required=True,  help="Pfad zur .h5 Datei")
parser.add_argument("--output",   type=str, required=True,  help="Zielordner")
parser.add_argument("--mode",     type=str, default="both", choices=["img", "both"],
                    help="'img' = nur Bild schneiden | 'both' = Bild + Maske")
parser.add_argument("--res",      type=int, default=256,    help="Tile-Größe (default: 256)")
parser.add_argument("--step",     type=int, default=None,   help="Schrittweite (default: res//2)")
args = parser.parse_args()

img_path  = Path(args.input)
out_path  = Path(args.output)
res_size  = args.res
step_size = args.step if args.step is not None else res_size // 2

# ── H5 laden ─────────────────────────────────────────────────────────────────
def load_array_from_h5(path, bounds="0 -1 0 -1", return_size=False):
    bounds = np.array(bounds.split(" ")).astype(float).astype(int)
    with h5py.File(path, "r") as f:
        key   = "Image" if "Image" in f.keys() else "exported_data"
        shape = f[key].shape
        if return_size:
            return shape
        if all([bounds[i] == [0, -1, 0, -1][i] for i in range(4)]):
            bounds = [0, shape[0], 0, shape[1]]
        img = np.zeros((bounds[1]-bounds[0], bounds[3]-bounds[2]))
        f[key].read_direct(img, (slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3])))
    return img

# ── Maske berechnen ───────────────────────────────────────────────────────────
class MaskLayers:
    def __init__(self, img_path, bounds):
        self.img_path  = img_path
        self.bounds    = bounds
        raw_path       = Path(img_path).parent.parent
        stem           = Path(img_path).stem
        suffix         = Path(img_path).suffix

        self.bckg_path   = raw_path / "bgr_mask"    / f"{stem}-Image_Probabilities_background{suffix}"
        self.vessel_path = raw_path / "vessel_mask"  / f"{stem}-Image_Probabilities_255_8b{suffix}"

    def image_data(self):
        return load_array_from_h5(self.img_path, self.bounds)

    def background_data(self):
        return load_array_from_h5(self.bckg_path, self.bounds)[:, :, 1]

    def vessel_data(self):
        return load_array_from_h5(self.vessel_path, self.bounds)[:, :, 0]

    def background_threshed(self):
        return np.where(self.background_data() >= 100, 1, 0)

    def only_largest_component(self):
        inp    = (self.background_threshed() > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inp, 4, cv2.CV_32S)
        areas  = stats[1:, cv2.CC_STAT_AREA]
        main   = np.argmax(areas) + 1
        return (labels == main).astype("uint8") * 255

    def vessel_threshed(self):
        return np.where(self.vessel_data() < 75, 0, 1)

    def mask(self):
        return np.multiply(self.only_largest_component(), self.vessel_threshed())

# ── Ausgabeordner anlegen ─────────────────────────────────────────────────────
img_dir  = out_path / "img"
img_dir.mkdir(parents=True, exist_ok=True)

if args.mode == "both":
    mask_dir = out_path / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

# ── Tiles schneiden ───────────────────────────────────────────────────────────
shape = load_array_from_h5(img_path, return_size=True)
h, w  = shape[0], shape[1]

print(f"Bild:    {img_path.name}")
print(f"Größe:   {h} x {w}")
print(f"Modus:   {args.mode}")
print(f"Tile:    {res_size}px  |  Schritt: {step_size}px")
print(f"Ziel:    {out_path}")

for i in tqdm(range(0, h - (res_size + h % res_size), step_size), desc="Zeilen"):
    for j in tqdm(range(0, w - (res_size + w % res_size), step_size), desc="Spalten", leave=False):
        bounds  = f"{i} {i+res_size} {j} {j+res_size}"
        tile    = MaskLayers(img_path, bounds)
        name    = f"{img_path.stem}_{i}_{j}.png"

        plt.imsave(img_dir / name, tile.image_data(), cmap="gray")

        if args.mode == "both":
            plt.imsave(mask_dir / name, tile.mask(), cmap="gray")

print("Fertig.")
