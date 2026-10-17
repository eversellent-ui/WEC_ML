#  EcoSort Starter Kit
### Intel SIG Recruitment Challenge — Edge AI Track

Welcome to the **EcoSort Edge-AI Challenge**! This starter kit gives you the minimal scaffolding needed to load the dataset, build your custom model, and export a submission-ready `.onnx` file without any boilerplate headaches.

The dataset can be found at the link: https://drive.google.com/file/d/12lDsBBYZHPM_a8vV2D4CRgxgPH0VI-ZZ/view?usp=sharing


## Structure of the kit

```
starter_kit/
├── requirements.txt      # Minimal Python dependencies
├── download_data.py      # 1-click dataset setup tool
├── train_template.py     # Clean skeleton for training & ONNX export
├── verify_submission.py  # Pre-submission verification tool (checks size & constraints)
└── README.md             # This guide
```

## 3-Step Quickstart

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Unpack the Dataset
```bash
python download_data.py
```
This extracts:
- `data/public/train/`: 900 training images (150 per class)
- `data/public/val/`: 240 validation images (40 per class)

### Step 3: Train & Export Your Model
```bash
python train_template.py --epochs 10 --batch_size 32 --lr 0.001
```
*Note: `train_template.py` contains a naive 2-layer toy network (~5k parameters) that demonstrates how the entire pipeline should work. You must replace it with your own architecture to achieve the required $\ge 75\%$ accuracy.*

### Step 4: Verify Before Submitting
Before uploading your model to the portal, run the local verification suite:
```bash
python verify_submission.py --model model.onnx --val_dir data/public/val
```
Ensure all technical constraints pass:
1. File size $\le 1.50\text{ MB}$
2. Total parameters $\le 350,000$
3. Input shape: `(Batch, 3, 96, 96)` float32
4. Output shape: `(Batch, 6)` float32

---

## The Challenge: Design Your Own Approach

This recruitment challenge is **open-ended**. You are free to design any architecture and training methodology you choose, provided it respects the hardware budget.


## Submitting Your Work

Once your `verify_submission.py` checks pass:
1. Open the Google Form at the link - https://docs.google.com/forms/d/e/1FAIpQLSdj3D2CX9UL6Veoi2nX1NsnWwtvHa4h-ehnIlrPhbCKElk-gw/viewform?usp=header
2. Enter your details (Full Name, Student Email, Roll Number, and public GitHub repository link with your training code).
3. Upload `model.onnx`. The file will be used to evaluate your model's performance on unseen data by the organizers.
