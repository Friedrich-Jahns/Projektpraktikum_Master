# tile_cutter.py

Schneidet große `.h5` Mikroskopbilder in kleinere Tiles (Kacheln) für das Training eines U-Net Segmentierungsmodells. Optional wird gleichzeitig die zugehörige Gefäßmaske ausgeschnitten.

---

## Voraussetzungen

```bash
pip install numpy matplotlib h5py opencv-python tqdm
```

---

## Verwendung

```bash
python tile_cutter.py --input <pfad_zur_h5> --output <zielordner> --mode <img|both>
```

### Argumente

| Argument | Pflicht | Standard | Beschreibung |
|---|---|---|---|
| `--input` | ✓ | – | Pfad zur `.h5` Eingabedatei |
| `--output` | ✓ | – | Zielordner für die Tiles |
| `--mode` | – | `both` | `img` = nur Bild / `both` = Bild + Maske |
| `--res` | – | `256` | Tile-Größe in Pixeln |
| `--step` | – | `res//2` | Schrittweite (Überlappung bei `step < res`) |

---

## Beispiele

```bash
# Nur Bildtiles (ungefärbter Datensatz, kein Mask-Pfad nötig)
python tile_cutter.py --input scan.h5 --output dat/ungefaerbt --mode img

# Bild + Maske (für Training)
python tile_cutter.py --input scan.h5 --output dat/train --mode both

# Größere Tiles, mehr Überlappung
python tile_cutter.py --input scan.h5 --output dat/train --mode both --res 512 --step 128
```


