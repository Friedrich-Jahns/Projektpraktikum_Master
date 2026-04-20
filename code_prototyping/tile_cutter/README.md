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

---

## Ausgabestruktur

```
<output>/
  img/       # Bildtiles als .png
  mask/      # Maskentiles als .png  (nur bei --mode both)
```

Bild und zugehörige Maske haben jeweils **denselben Dateinamen** (`<stem>_<i>_<j>.png`), sodass sie direkt vom Dataloader geladen werden können.

---

## Erwartete Ordnerstruktur der Eingabedaten

Die Masken werden automatisch relativ zur `.h5` Datei gesucht:

```
data/
  raw/
    img/
      scan.h5                         ← Eingabebild
    bgr_mask/
      scan-Image_Probabilities_background.h5
    vessel_mask/
      scan-Image_Probabilities_255_8b.h5
```

---

## Maskenberechnung

Die finale Maske kombiniert zwei Schritte:

1. **Hintergrundmaske** – Gewebebereich wird per Schwellwert (`>= 100`) und größter zusammenhängender Komponente (Connected Components) isoliert
2. **Gefäßmaske** – Gefäßwahrscheinlichkeit wird per Schwellwert (`>= 75`) binarisiert

Beide werden multipliziert → Gefäße **nur innerhalb** des Gewebes.
