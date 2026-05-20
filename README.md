# Car Damage Analyzer

Computer vision project for estimating car damage severity from a photo. Created for "[ВНЕДРЕЙД](https://www.orionsoft.ru/events/vnedreid-nbspxakatonnbspotnbsporionnbspsoft)" hackathon 07-08.06, 2025.

<img width="1303" height="959" alt="image_2025-06-08_17-09-00 (2)" src="https://github.com/user-attachments/assets/3ba4d168-b9d6-4af0-ac26-ba85ab62e5e2" />


The current pipeline uses two semantic segmentation models:

- damage model: predicts 9 damage classes
- parts model: predicts 13 grouped car-part classes

The final severity grade is computed by intersecting predicted damage masks with predicted part masks and applying a fixed damage/part weight matrix. Public grades are clamped to the `1..5` range.

Participants:

- Oleg Ryabinin

- Kirill Koltsov

- Artem Filippov

From Graphics & Media Lab, Moscow State University.

## Repository Layout

```text
car_damage_cv/        Reusable package: models, preprocessing, inference, scoring
scripts/predict.py    CLI inference for a local image or URL
scripts/train.py      Generic U-Net segmentation training script
app.py                Flask demo API/UI
templates/, static/   Flask frontend
notebooks/            Original exploratory notebooks
tools/                One-off dataset helper scripts
grade.py, metric.py   Compatibility wrappers for older notebooks
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place trained checkpoints in the project root by default:

```text
model_damage.pth
model_parts.pth
```

Checkpoints should contain either a raw PyTorch state dict or a dict with `model_state_dict`.

## Inference

Run prediction from a local file:

```bash
python scripts/predict.py --image path/to/car.jpg
```

Run prediction from an image URL:

```bash
python scripts/predict.py --url "https://example.com/car.jpg" --json
```

Use custom checkpoints:

```bash
python scripts/predict.py \
  --image path/to/car.jpg \
  --damage-checkpoint checkpoints/damage.pth \
  --parts-checkpoint checkpoints/parts.pth
```

## Web Demo

```bash
python app.py
```

Then open `http://localhost:5000`.

Optional environment variables:

```bash
DAMAGE_MODEL_PATH=checkpoints/damage.pth
PARTS_MODEL_PATH=checkpoints/parts.pth
DEVICE=cuda
IMAGE_SIZE=512
PORT=5000
```

## Training

`scripts/train.py` trains one segmentation model at a time. Use it once for damage masks and once for parts masks.

Expected data layout:

```text
data/
  damage/
    images/
      0001.jpg
    masks/
      0001.png
  parts/
    images/
      0001.jpg
    masks/
      0001.png
```

Masks must be single-channel class-index images where each pixel value is the class id.

Train the damage model:

```bash
python scripts/train.py \
  --train-images data/damage/images \
  --train-masks data/damage/masks \
  --num-classes 9 \
  --output checkpoints/model_damage.pth \
  --epochs 30 \
  --batch-size 4
```

Train the parts model:

```bash
python scripts/train.py \
  --train-images data/parts/images \
  --train-masks data/parts/masks \
  --num-classes 13 \
  --output checkpoints/model_parts.pth \
  --epochs 30 \
  --batch-size 4
```

If pretrained encoder weights are not available locally, pass `--encoder-weights none`.

## Notes

- Inference resizes and pads input images to `512x512` by default.
- The scoring constants live in `car_damage_cv/constants.py`.
- `grade.py`, `metric.py`, and `utils.py` remain as thin compatibility wrappers for old notebook imports.
