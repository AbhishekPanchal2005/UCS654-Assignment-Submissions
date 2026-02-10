# GAN-based PDF Estimation Assignment

## Student Information
- **Name:** Abhishek Panchal
- **Roll Number:** 102317167
- **Date:** 10 February 2026

## Assignment Objective
Learn an unknown probability density function of a transformed random variable using a Generative Adversarial Network (GAN).

---

## Implementation

### Step 1: Data Transformation

**Dataset:** India Air Quality Data (NO₂ concentration)
- Source: Kaggle India Air Quality Dataset
- Feature: NO₂ concentration values

**Transformation Function:**
```
z = x + ar * sin(br * x)
```

**Transformation Parameters:**
- Roll Number (r): 12345
- ar = 0.5 * (r mod 7) = 0.5 * (12345 mod 7) = 0.5 * 1 = **0.5**
- br = 0.3 * (r mod 5 + 1) = 0.3 * (12345 mod 5 + 1) = 0.3 * 3 = **0.9**

**Final Transformation:**
```
z = x + 0.5 * sin(0.9 * x)
```

---

### Step 2: GAN Architecture

#### Generator Network
- **Input:** Random noise vector (dimension: 20)
- **Architecture:**
  - Linear(20 → 48) + LeakyReLU(0.2)
  - Linear(48 → 24) + LeakyReLU(0.2)
  - Linear(24 → 1)
- **Output:** Generated sample z_fake

#### Discriminator Network
- **Input:** Real or generated sample (dimension: 1)
- **Architecture:**
  - Linear(1 → 24) + LeakyReLU(0.2) + Dropout(0.3)
  - Linear(24 → 12) + LeakyReLU(0.2) + Dropout(0.3)
  - Linear(12 → 1) + Sigmoid
- **Output:** Probability [0,1] indicating real vs fake

#### Training Configuration
- **Optimizer:** Adam
  - Generator learning rate: 0.0015
  - Discriminator learning rate: 0.0003
  - Beta parameters: (0.5, 0.999)
- **Loss Function:** Binary Cross Entropy (BCE)
- **Batch Size:** 512
- **Epochs:** 50
- **Label Smoothing:** Real=0.95, Fake=0.05

---

### Step 3: PDF Estimation

After training, generated 50,000 samples from the trained generator and estimated the probability density using:
1. **Histogram density estimation** (80 bins)
2. **Kernel Density Estimation (KDE)** using Gaussian kernel

---

## Results

### Statistical Comparison

| Metric | Real Data | Generated Data |
|--------|-----------|----------------|
| Mean   | 25.80     | 30.81          |
| Std Dev| 18.52     | 3.50           |
| Min    | 0.00      | 20.33          |
| Max    | 876.07    | 52.51          |

### Training Metrics
- **Final Generator Loss:** 0.697
- **Final Discriminator Loss:** 1.396
- **Training Time:** ~2 minutes (50 epochs)

---

## Observations and Analysis

### 1. Mode Coverage
The GAN exhibited **mode collapse**, a well-documented challenge in GAN training:
- Generated samples concentrated in narrow range (20-52) vs real data (0-876)
- Generator captured the central tendency but failed to learn the full distribution
- Missing tail regions and extreme values completely

### 2. Training Stability
- Losses plateaued near theoretical equilibrium values (ln(2)≈0.693, ln(4)≈1.386)
- Indicates Nash equilibrium but not optimal distribution learning
- Discriminator and generator reached balance but at suboptimal point
- Some fluctuation observed around epoch 40-50 suggesting potential for improvement with longer training

### 3. Quality of Generated Distribution
**Strengths:**
- Successfully learned approximate location (mean: 30.8 vs 25.8)
- Generated PDF follows similar shape to real PDF near the peak
- Samples are plausible values within the NO₂ concentration range

**Limitations:**
- Severe underestimation of variance (std: 3.5 vs 18.5)
- Failed to capture heavy-tailed distribution
- Mode collapse prevented diversity in generated samples

---

## Lessons Learned

1. **GAN Training Difficulty:** Experienced firsthand why GANs are considered difficult to train - mode collapse, vanishing gradients, and equilibrium challenges

2. **Importance of Hyperparameters:** Small changes in learning rates significantly impact training dynamics

3. **Trade-offs:** Balancing discriminator strength vs generator capacity requires careful tuning

4. **Alternative Approaches:** For 1D distribution learning, simpler methods (mixture models, kernel density estimation, normalizing flows) might be more appropriate

---

## Future Improvements

If given more computational resources and time, the following improvements could be explored:

1. **Extended Training:** 200-500 epochs with learning rate scheduling
2. **Architecture Modifications:**
   - Deeper networks with residual connections
   - Batch normalization in generator
   - Spectral normalization in discriminator
3. **Advanced GAN Variants:**
   - Wasserstein GAN (WGAN) for better stability
   - Progressive GAN for gradual complexity increase
4. **Alternative Objectives:**
   - Maximum Mean Discrepancy (MMD) loss
   - Energy-based GAN
5. **Ensemble Approach:** Multiple generators to capture different modes

---

## Conclusion

This assignment provided valuable hands-on experience with Generative Adversarial Networks. While the GAN did not perfectly learn the target distribution due to mode collapse, the implementation successfully demonstrated:

- Understanding of GAN architecture and adversarial training
- Proper data transformation and preprocessing
- PDF estimation using generated samples
- Critical analysis of model limitations

The challenges encountered illustrate why GAN research remains an active area - training stability and mode coverage continue to be fundamental problems even in simple settings like 1D distribution learning.

---

## Code Files

- `a2_gan.py` - Main implementation file
- `gan_pdf_results.png` - Visualization of results
