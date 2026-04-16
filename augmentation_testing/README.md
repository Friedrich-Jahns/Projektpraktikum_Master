# Befehle

## Training

```bash
python run.py --run_name <name> --augmentation <aug>
```

### Optionale Parameter

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--augmentation` | `baseline` | Augmentierung aus `aug/` |
| `--run_name` | – (Pflicht) | Name des Experiments |
| `--epochs` | `100` | Anzahl Epochen |
| `--bs` | `8` | Batch Size |
| `--lr` | `1e-3` | Learning Rate |

## Beispiele

```bash
# Baseline
python run.py --run_name baseline --augmentation baseline

# Neue Augmentierung
python run.py --run_name stain_aug --augmentation stain_gray

# Mit angepassten Parametern
python run.py --run_name test --augmentation stain_gray --epochs 50 --bs 4 --lr 5e-4
```

## Ergebnisse

Jeder Run speichert in `res/<run_name>/`:

```
res/<run_name>/
├── best_model.pth
├── last_model.pth
├── train_log.png
├── train_log.npy
└── config.json
```

## Neue Augmentierung hinzufügen

1. Datei in `aug/` anlegen, z.B. `aug/meine_aug.py`
2. Funktion `get_augmentation()` implementieren die `(img, mask) -> (img, mask)` zurückgibt
3. Mit `--augmentation meine_aug` aufrufen
