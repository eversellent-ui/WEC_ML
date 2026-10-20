# EcoSortCNN: Edge-AI Waste Classifier

Intel SIG Recruitment Challenge 2026 (EcoSort). A compact CNN that classifies 96×96 RGB waste images into 6 classes (cardboard, glass, metal, paper, plastic, trash), trained from scratch under strict edge-hardware limits.

**Headline result (exported `model.onnx`, checked with the organizers' `verify_submission.py`):**

| Check | Result | Limit |
|---|---|---|
| File size | 0.92 MB | ≤ 1.50 MB |
| Parameters (counted from ONNX) | 241,606 | ≤ 350,000 |
| Input / output | `(batch, 3, 96, 96)` → `(batch, 6)`, float32 | same |
| Val accuracy (240 images) | 100.00% | hidden test ≥ 75% |
| CPU latency | 3.35 ms/image | ≤ 35 ms (soft target) |

> **Caveat:** the validation set is small (240 images, so one image is about 0.4%), and I also used it to pick the exported checkpoint (lowest validation loss). 100% on validation is therefore optimistic, and the hidden-test score may be lower.

---

## 1. Problem and constraints

- Task: 6-class image classification of recyclable waste.
- Hard constraints: valid ONNX file, ≤ 350,000 parameters, ≤ 1.50 MB, ≥ 75% accuracy on a hidden test set.
- Training hardware: laptop with an NVIDIA RTX 5050.
- Environment (Python 3.14): matplotlib 3.11.2, numpy 2.5.2, onnx 1.23.2, onnxruntime 1.31.0, pillow 12.3.0, scikit-learn 1.9.1, scipy 1.18.1, torch 2.14.1+cu130, torchvision 0.29.1+cu130, tqdm 4.70.1.

## 2. Approach in brief

The dataset is small (900 training images), so overfitting was the main risk. I started from the organizers' toy classifier, worked out why it was too weak, and replaced it with a deeper convolutional network using BatchNorm, global average pooling and dropout. I trained it with augmentation, AdamW weight decay and a cosine learning-rate schedule, and exported the epoch with the lowest validation loss.

**Why the toy model was weak.** It reached about 35% accuracy with roughly 5,500 parameters, a small fraction of the 350k budget. It had few channels (16 and 32) and applied 4×4 max-pooling twice, which discards spatial detail very quickly.

## 3. Data

- 900 training images (150 per class) and 240 validation images (40 per class). The classes are balanced, so accuracy is not distorted by one class dominating.
- The original images were already 96×96.
- Preprocessing: resize to 96×96, scale to 0-1, normalize with ImageNet mean/std. This matches the verifier, so the exported model expects exactly that input.
- **Augmentation used:** `RandomHorizontalFlip` and `RandomRotation`, applied to the training set only. A flipped or rotated item is still the same item, and objects appear at different orientations in real photos. Validation has no augmentation so that it measures performance on unmodified images.
- **Augmentation not used:** colour jitter and cropping. The images contain simple shapes with well-defined colours, and hue changes could have confused the model by removing the colour cues that separate materials.

## 4. Architecture (`Model_A.py`, class `Material_Identifier`)

Each stage is Convolution → BatchNorm → ReLU → Pooling. After the last convolution, global average pooling (`AdaptiveAvgPool2d`) reduces each feature map to one number, followed by dropout (0.3) and one Linear layer producing 6 logits.

| Layer | Shape | Parameters |
|---|---|---|
| Conv (features.0) | 3 → 32, 3×3, no bias | 864 |
| Conv (features.4) | 32 → 64 | 18,432 |
| Conv (features.8) | 64 → 128 | 73,728 |
| Conv (features.12) | 128 → 128 | 147,456 |
| BatchNorm ×4 (weight + bias) | 32, 64, 128, 128 channels | 704 |
| Linear (classifier.1) | 128 → 6 | 774 |
| **Total (PyTorch, trainable)** | | **241,958** |


A conv layer has `C_out × C_in × k × k` weights, so the later layers cost far more than the early ones even with the same 3×3 kernel: both `C_in` and `C_out` grow. The last convolution alone holds about 61% of all parameters.

Spatial size after each pooling stage: 96×96 → 48×48 → 24×24 → 12×12 → 6×6.

### Why the ONNX count (241,606) differs from PyTorch (241,958)

So The Manual Calculation gives the parameter count exactly equal to what Onxx count is i.e 241606. Here's how i got this
In CNN's the formula for parameter calculation is Cin*Cout*k*k + Bias(No. of neurons in Output i.e. =Cout) 
So the no. of features go this way (3->32->64->128->128 i.e. through convolutions), the kernel size is 3 
    3*32*9 + 32
    32*64*9 +64
    64*128*9 +128
    128*128*9 +128
    128*1 + 6  [ this is the last layer where there are 128 features of 6*6 size and global average pooling is applied leading to a vector of 128 individual scalars so here kernels have no effect] and 6 is the output which is the 6 different classes. 
The difference is exactly 352, which equals the total number of BatchNorm channels (32 + 64 + 128 + 128). When the model is exported in eval mode, each BatchNorm is folded into the convolution before it. The convolutions have no bias of their own, so folding adds one bias per channel (+352) and removes the BatchNorm weight and bias (−704): 241,958 − 704 + 352 = 241,606.

### Why global average pooling (GAP)

GAP averages each feature map into a single number, so the final Linear layer receives 128 inputs and has 128 × 6 + 6 = 774 parameters. Without GAP, flattening the final 6×6 grids of 128 channels would give 4,608 inputs and 27,654 parameters, about 36 times more. The without-GAP size would also grow with the image: at 192×192 the final grid would be 12×12 and the layer about four times larger, while the GAP layer stays at 774. The cost is that GAP discards position information, which is acceptable here because a glass bottle is still glass wherever it sits in the photo. Source: *Network in Network*, read through d2l.ai.

### ReLU

`f(x) = max(0, x)`. I chose it over sigmoid because sigmoid saturates: its derivative is at most 0.25 and becomes almost zero for large |z|, so gradients shrink and learning slows. ReLU's gradient is 1 for positive inputs, and it is also cheaper to compute.

## 5. Training setup

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR |
| Dropout | 0.3 |
| Batch size | 32 |
| Epochs | 15 |
| Loss | Cross-entropy |
| Seed | none set, so runs are not exactly reproducible |
| Checkpoint exported | epoch with the lowest validation loss |

**Why track validation loss, not only accuracy.** Two models can have the same accuracy but very different loss. A model that says "glass 0.52 / plastic 0.48" and one that says "glass 0.93 / plastic 0.07" are both wrong if the image is plastic, but the second is far more confidently wrong, and only the loss shows that. So I save the epoch with the lowest validation loss.

**Overfitting.** It happens when the dataset is small and the parameter count is large, so the model has enough capacity to memorize the training images. I counter it with augmentation (more varied training examples), dropout (30% of neurons are dropped during training and all are used at evaluation), and weight decay.

**Weight decay.** At each step, AdamW shrinks every weight slightly toward zero, separately from the gradient update. This penalizes large weights and encourages simpler models. It does not bound the weights to a range such as 0-1, and it is different from normalization.

## 6. Experiments

I did not keep a systematic log of every change, so this table is informal and incomplete.

| # | Change | Result | What I learned |
|---|---|---|---|
| 0 | Starter toy model | about 35% val, about 5.5k params | far too small and too aggressive in pooling |
| 1 | New 4-conv CNN with BN + GAP | see training log | |
| 2 | Learning rate 0.003 | 45% after epoch 1 | too high |
| 3 | Learning rate 0.002 | 56% after epoch 1 | better, still slow |
| 4 | Learning rate 0.001 | 87% after epoch 1 | best of the three, kept for the final model |
| 5 | Confusion matrix using `argmin` instead of `argmax` | all-zero diagonal | not valid evidence (see Section 7) |

Differences of about 1 point on a 240-image validation set are within noise.

**Training log (final run):**

```
Epoch [01/15] Train Acc: 83.1% | Val Acc: 89.6%
Epoch [02/15] Train Acc: 97.2% | Val Acc: 98.3%
Epoch [03/15] Train Acc: 98.7% | Val Acc: 98.8%
Epoch [04/15] Train Acc: 98.3% | Val Acc: 98.8%
Epoch [05/15] Train Acc: 98.8% | Val Acc: 98.3%
Epoch [06/15] Train Acc: 99.0% | Val Acc: 99.6%
Epoch [07/15] Train Acc: 99.0% | Val Acc: 100.0%
Epoch [08/15] Train Acc: 97.6% | Val Acc: 99.6%
Epoch [09/15] Train Acc: 98.6% | Val Acc: 98.3%
Epoch [10/15] Train Acc: 99.2% | Val Acc: 98.3%
Epoch [11/15] Train Acc: 99.0% | Val Acc: 93.3%
Epoch [12/15] Train Acc: 99.7% | Val Acc: 98.8%
Epoch [13/15] Train Acc: 99.3% | Val Acc: 98.8%
Epoch [14/15] Train Acc: 99.4% | Val Acc: 100.0%
Epoch [15/15] Train Acc: 99.7% | Val Acc: 100.0%
```

## 7. Results and diagnosis

- Final exported model: 100.00% on the 240 validation images (via ONNX Runtime).
- **Confusion matrix** (`argmax` predictions; rows are true classes, columns are predicted classes, and each cell counts the images in that pair): 40 on every diagonal cell and 0 elsewhere, so no confusions on validation.

  ![Confusion matrix](docs/confusion_matrix.png)

- **The `argmin` experiment.** Out of curiosity I also built the matrix with `argmin`, which records the class the model considers *least* likely. Its diagonal was all zeros, which I first read as proof of a perfect model. That conclusion is wrong: it only says the true class is never the model's *least* likely guess, which a much weaker model would also satisfy. The `argmax` matrix is the meaningful one.
- Because there were no confusions on validation, there are no errors to analyze, and a larger or harder test set would be needed to see which classes the model mixes up.
- Validation accuracy fluctuates (93.3% at epoch 11), and it was also used for checkpoint selection, so it is not a true test score.

## 8. Constraint verification

Output of `verify_submission.py --model model.onnx --val_dir Template_Content/ESCORT_DATA_SET/val`:

```
1. Model File Size:    0.92 MB / 1.50 MB limit -> PASS [OK]
2. Total Parameters:   241,606 / 350,000 limit -> PASS [OK]
3. Input Signature:    Name='input', Shape=['batch_size', 3, 96, 96]
4. Output Signature:   Name='output', Shape=['batch_size', 6]
   Shape Validation:   PASS [OK]
5. Avg CPU Latency:    3.35 ms/image (Target: <= 35 ms)
6. Local Val Accuracy: 100.00% (240 samples)
ALL CONSTRAINT CHECKS PASSED
```

## 9. What came from the starter kit vs my work

**From the organizers:** the argparse setup, the training-loop skeleton, the ONNX export function, `verify_submission.py`, `download_data.py`, and the dataset.

**Mine:** the `Material_Identifier` architecture in `Model_A.py`, the augmentation, AdamW with weight decay, dropout, the cosine learning-rate schedule, best-checkpoint saving by validation loss, the confusion-matrix analysis, the learning-rate tuning, and this documentation.

## 10. Resources and AI usage

**Resources I used:**
- Michael Nielsen, *Neural Networks and Deep Learning*, Chapter 1.
- Howard and Gugger, *Deep Learning for Coders with fastai and PyTorch*, for the training workflow and overfitting.
- *Network in Network*, read through d2l.ai, for global average pooling.
- PyTorch and scikit-learn documentation.

**How I used AI.** I read about the concepts first and used AI to explain the difficult ones, and to give me function signatures and syntax for things I had already understood (for example, after learning how a confusion matrix is built, I asked for the signatures to code it). I then wrote the code and read the docs for the calls involved. The matplotlib visualization code (plots, parameter chart, confusion-matrix image) was written with AI assistance. The architecture, experiments and results are my own work. I Also Used AI to convert a human typed i.e. first i typed the entire Readme file then asked AI for proper formatting and language improvement.

## 11. How to reproduce

```
pip install -r requirements.txt
python Template_Content/starter_kit/train_template.py --data_dir Template_Content/ESCORT_DATA_SET --epochs 15 --batch_size 32 --lr 0.001 --output model.onnx
python Template_Content/starter_kit/verify_submission.py --model model.onnx --val_dir Template_Content/ESCORT_DATA_SET/val
```

The dataset is not in the repo and must be placed at the path above. `Model_A.py` must be importable by the training script. Results can differ slightly between runs because no random seed is set.
