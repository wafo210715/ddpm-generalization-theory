# Research Methodology Design

## Diffusion Models as Epistemic Probes for Urban Vitality Manifold Learning

**Date**: 2025-02-08
**Research Focus**: Identifying the "generalization window" in diffusion model training for urban vitality research

---

## Implementation Notes

### Hardware Configuration

**Platform**: Windows 10 with dual RTX 3090 (24GB each, NVLink connected)
- GPU 0: 3x NVLink @ 14.062 GB/s + PCIe Gen 4 x16 @ ~32 GB/s
- GPU 1: 3x NVLink @ 14.062 GB/s + PCIe Gen 4 x16 @ ~32 GB/s
- **Total interconnect**: ~74 GB/s per GPU

### Dual-GPU Training with PyTorch DataParallel

**Implementation Choice**: We use PyTorch's DataParallel for dual-GPU training.

**Checkpoint Format Difference**:
- **Single GPU**: Full DDPM state dict
  ```python
  {
    'beta': [...],           # Noise schedule
    'alpha': [...],          # 1 - beta
    'alpha_bar': [...],      # Cumprod(alpha)
    'eps_model': {...}       # U-Net weights
  }
  ```
- **Dual GPU**: U-Net weights only
  ```python
  {
    'conv1.weight': [...],   # Direct U-Net weights (no wrapper)
    'conv1.bias': [...],
    ...
  }
  ```

**Why the difference?**
- PyTorch DataParallel wraps the model: `eps_model = DataParallel(eps_model, device_ids=[0,1])`
- The wrapper has a `.module` attribute containing the actual model
- Our `save_model()` function unwraps before saving:
  ```python
  if was_dual_gpu and isinstance(ddpm.eps_model, DataParallel):
      torch.save(ddpm.eps_model.module.state_dict(), path)  # U-Net only
  else:
      torch.save(ddpm.state_dict(), path)  # Full DDPM
  ```
- **Rationale**: Unwrapped checkpoints are more portable and don't require DataParallel context to load

**Sampling Compatibility**:
Our `sample.py` automatically detects and handles both formats:
```python
checkpoint = torch.load(path)
if "eps_model" in checkpoint or "beta" in checkpoint:
    ddpm.load_state_dict(checkpoint)      # Full DDPM
else:
    ddpm.eps_model.load_state_dict(checkpoint)  # U-Net only
```

### Schedule Comparison Training Strategy

**Chosen Approach**: Parallel single-GPU training (Option 2)

**Rationale**:
- **Faster overall**: Two schedules train simultaneously (70 min vs 80 min sequential)
- **Better utilization**: Both GPUs work independently
- **Simpler analysis**: No interleaved logs to disentangle
- **Scalability**: Approach works even with different numbers of GPUs

**Trade-offs**:
- Per-session slower: Single GPU (~70 min) vs dual GPU (~40 min)
- But overall faster due to parallelization: 70 min vs 80 min

**Implementation**:
```bash
python train_parallel_schedules.py --schedule1 linear --schedule2 cosine --steps 10000
```
- Schedule 1 runs on GPU 0 with `CUDA_VISIBLE_DEVICES=0`
- Schedule 2 runs on GPU 1 with `CUDA_VISIBLE_DEVICES=1`
- Outputs go to `model_ckpts/schedule_comparison/{schedule_type}/`

**Note**: This is a PyTorch-specific implementation choice. Other frameworks (JAX, TensorFlow) may handle multi-GPU training differently.

---

This document outlines the methodological rationale for using **discrete-timestep diffusion models** to study how neural networks learn and represent the low-dimensional manifold of urban vitality. The choice of discrete over continuous-time formulations is driven by our primary research goals: (1) identifying when memorization occurs during training, and (2) proving that diffusion models learn low-dimensional structural concepts before memorizing idiosyncratic noise.

---

## Research Goals

### Goal 1: Finding the "Generalization Window"

**Research Question**: At what point during training does a diffusion model switch from learning structural concepts of urban vitality to memorizing specific training examples?

**Methodological Requirements**:
- Track model behavior at discrete training milestones
- Compute memorization metrics (k-NN gap ratio, FID) at specific checkpoints
- Analyze how memorization evolves over training progress

**Why Discrete Timesteps?**
- Training progress is naturally discrete: step 100, 250, 500, 1000, 5000...
- Checkpoints map directly to training iterations
- "When does memorization begin?" has a clear discrete answer

### Goal 2: Proving Low-Dimensional Manifold Learning

**Research Question**: Do diffusion models learn the intrinsic low-dimensional structure of urban vitality data, or do they simply memorize high-dimensional pixel correlations?

**Methodological Requirements**:
- Jacobian rank analysis at multiple training stages
- Spectral density analysis of learned parameters
- Demonstrate that denoising autoencoder remains low-rank throughout training

**Why Discrete Timesteps?**
- Can analyze manifold properties at specific timesteps (0, 250, 500, 1000...)
- Easier to connect "what is learned at step 500" vs "what is learned at step 5000"
- Sufficient for theoretical validation of low-rank properties

### Goal 3: Latent Space as "Computational Laboratory" (Future Work)

**Research Question**: Can we navigate the diffusion model's latent space to uncover the "hidden generative grammar" of urban vitality?

**Methodological Requirements**:
- Deterministic sampling trajectories (DDIM)
- Interpolation between concepts in latent space
- Time-step-specific analysis of what features are learned

**Note**: Continuous-time formulations have advantages here, but DDIM on discrete timesteps provides similar benefits.

---

## Theoretical Background: Discrete vs. Continuous Time Formulations

### Discrete-Time Diffusion (DDPM)

**Mathematical Formulation**:
- Time variable: t ∈ {0, 1, 2, ..., T} where T = 1000
- Forward process: q(x_t | x_{t-1}) - discrete Markov chain
- Noise schedule: β_t (discrete values, one per timestep)
- Parameterization: ε_θ(x_t, t) predicts noise at discrete timestep t

**Training**:
```python
# Sample discrete timestep uniformly
t ~ Uniform({0, 1, ..., 999})
# Predict noise at that timestep
ε_θ = model(x_t, t)
```

**Key Properties**:
- Each timestep represents a specific noise level
- Forward process requires T steps to reach pure noise
- Sampling requires going through all timesteps (or DDIM for acceleration)

### Continuous-Time Diffusion (EDM / Score-Based SDEs)

**Mathematical Formulation**:
- Time variable: t ∈ [0, 1] continuous
- Forward process: dx = f(x,t)dt + g(t)dw (Stochastic Differential Equation)
- Noise schedule: σ(t) (continuous function)
- Parameterization: D_θ(x, σ) predicts denoised image at noise level σ

**Training**:
```python
# Sample continuous time uniformly
t ~ Uniform([ε_t, 1])  # where ε_t = 1e-5
# Convert to noise level
σ = σ(t)
# Predict denoised image
D_θ = model(x, σ)
```

**Key Properties**:
- Time represents continuous noise progression
- Forward process is an SDE with infinite "steps"
- Can use ODE solvers for deterministic sampling
- Theoretical connection to score-based generative modeling

---

## Methodological Decision: Why Discrete Timesteps?

### Rationale for Our Research

| Aspect | Discrete Timesteps | Continuous Time | Decision |
|--------|-------------------|-----------------|----------|
| **Training milestones** | Natural mapping to checkpoints (100, 500, 1000 steps) | Training iterations ≠ continuous time | **Discrete** ✓ |
| **Memorization tracking** | Direct: "memorization begins at step ~5000" | Indirect: need to map iterations to σ | **Discrete** ✓ |
| **Jacobian analysis** | Works at each discrete timestep | Works at any σ | **Both OK** |
| **Theoretical elegance** | Simpler implementation | More elegant SDE connection | **Discrete** (pragmatism) |
| **Latent space navigation** | DDIM provides deterministic sampling | Native ODE solvers | **Discrete** (DDIM sufficient) |
| **Computational efficiency** | Standard DDPM, widely used | Slightly more complex | **Discrete** ✓ |

### Key Insight: Schedules vs. Time Formulation

**Critical distinction**: The **schedule type** (linear, cosine, VP, VE) is independent of the **time formulation** (discrete vs. continuous).

We can use VP/VE **schedules** with discrete timesteps by discretizing the continuous formulas:

| Schedule | Native Time | Can be Discretized? | Our Approach |
|----------|-------------|---------------------|--------------|
| Linear | Both | ✓ (native) | Discrete |
| Cosine | Both | ✓ (native) | Discrete |
| VP (Variance Preserving) | Continuous | ✓ (discretize σ → β) | **Discrete VP** |
| VE (Variance Exploding) | Continuous | ✓ (discretize σ → β) | **Discrete VE** |
| EDM | Continuous | ✓ (convert σ → β) | Future work |

This gives us the benefits of different noise schedules while maintaining the discrete training framework that aligns with our research goals.

---

## Implementation Strategy

### Phase 1: Discrete Foundation (Current)

**Completed**:
- ✅ Environment setup with dual RTX 3090 GPUs
- ✅ DDPM implementation with discrete timesteps (T=1000)
- ✅ Logarithmic checkpointing: 100, 250, 500, 1000, 2500, 5000, 10000, 20000...
- ✅ Config-based training system

**Current Status**:
- Discrete timesteps: t ∈ {0, 1, ..., 999}
- Schedule: Linear (β from 1e-4 to 2e-2)
- Training: 50,000 steps on MNIST (baseline validation)

### Phase 2: Schedule Variants (In Progress)

**Goal**: Compare how different noise schedules affect memorization dynamics

**Implementation**:
1. **Linear schedule** (baseline): β_t = β₀ + (β_T - β₀) · t/T
2. **Cosine schedule** (improved signal preservation): Based on Nichol & Dhariwal 2021
3. **VP schedule** (discretized): Convert σ(t) = √(exp(0.5·β_d·t² + β_min·t) - 1) to discrete β_t
4. **VE schedule** (discretized): Convert σ(t) = σ_min·(σ_max/σ_min)^t to discrete β_t

**Discretization Method**:
```python
# For VP/VE schedules, we compute σ at each discrete timestep
# Then convert to β using the relationship:
# α̅_t = 1 / (1 + σ²)
# β_t = 1 - α_t = 1 - (α̅_t / α̅_{t-1})
```

**Research Question**: Does the noise schedule affect when the "generalization window" closes?

### Phase 3: Memorization Tracking (Planned)

**Metrics**:
1. **k-NN gap ratio** (fraction collapsed): Measures memorization
2. **FID (Fréchet Inception Distance)**: Measures sample quality
3. **Reproducibility score** (SSCD): Measures consistency across samples

**Analysis**:
- Plot memorization fraction vs. training steps
- Identify "generalization window": training range where memorization is low but quality is high
- Compare memorization dynamics across schedules

### Phase 4: Manifold Analysis (Planned)

**Methods**:
1. **Jacobian rank analysis**: Prove denoising autoencoder is low-rank
2. **Spectral density analysis**: Analyze learned parameter distributions
3. **Intrinsic dimensionality estimation**: Find minimal data dimension

**Validation**:
- Show Jacobian rank remains low throughout training
- Demonstrate model learns intrinsic data dimensions, not pixel correlations
- Compare with theoretical predictions from diffusion literature

---

## Connection to Research Repositories

### Repository 1: Why Diffusion Models Don't Memorize

**Approach**: Discrete timesteps (DDPM, 1000 steps)

**Key Finding**: Implicit dynamical regularization during diffusion training prevents memorization

**Relevance to Our Work**:
- ✅ Uses discrete timesteps to track memorization
- ✅ Logarithmic checkpointing strategy
- ✅ k-NN gap ratio for memorization measurement
- ✅ Fixed-time training mode (train at specific timestep only)

**Our Adaptation**:
- Same discrete framework
- Focus on urban vitality data (geo-spatial images)
- Extend to multiple schedule types (VP/VE)

### Repository 2: Diffusion Model Generalizability

**Approach**: Continuous time (EDM, SDE-based)

**Key Finding**: Diffusion models learn low-dimensional distributions via subspace clustering

**Relevance to Our Work**:
- Theoretical foundation for manifold learning
- Jacobian rank analysis methods
- SSCD (self-supervised descriptors) for reproducibility

**Our Adaptation**:
- Discretize continuous schedules for our discrete framework
- Use Jacobian analysis to validate low-rank properties
- Apply subspace clustering concepts to urban vitality data

---

## Methodological Justification Summary

### Why This Approach is Appropriate for Urban Vitality Research

1. **Urban vitality as a low-dimensional phenomenon**
   - Cities exhibit structural patterns (streets, buildings, public spaces)
   - These patterns exist on a low-dimensional manifold
   - High-dimensional pixel data contains redundant information

2. **Diffusion models as "structure-preserving compressors"**
   - Training forces model to learn efficient representations
   - Early training: learn structural concepts (low-rank)
   - Late training: memorize noise (full-rank)
   - "Generalization window" = sweet spot between structure and overfitting

3. **Discrete timesteps align with our research questions**
   - "When does memorization occur?" → Discrete step number
   - "What is learned at each stage?" → Analyze checkpoint at step N
   - "How do schedules affect memorization?" → Compare discrete training curves

4. **Geo-spatial data considerations**
   - ArcGIS projections → TIF → JPG (512×512 or 1024×1024)
   - High resolution but potentially low intrinsic dimension
   - Discrete framework handles large images efficiently

---

## Limitations and Future Directions

### Current Limitations

1. **Discrete timesteps may miss fine-grained dynamics**
   - Continuous time offers smoother trajectories
   - Mitigation: High-resolution checkpointing (logarithmic)

2. **Schedule comparison may be confounded by other factors**
   - Learning rate, batch size, architecture also affect memorization
   - Mitigation: Control experiments, only vary schedule type

3. **Geo-spatial data may require different hyperparameters**
   - Urban imagery differs from MNIST/CIFAR
   - Mitigation: Iterative hyperparameter tuning

### Future Directions

1. **Continuous-time analysis** (Goal 3)
   - Implement EDM-style continuous training
   - Compare discrete vs. continuous on same data
   - Theoretical analysis of SDE dynamics

2. **Higher-resolution scaling**
   - Scale from 256×256 to 512×512 to 1024×1024
   - Memory-efficient training with gradient checkpointing
   - Multi-GPU optimization

3. **Urban-specific architectures**
   - Incorporate spatial attention mechanisms
   - Multi-scale processing for different urban features
   - Conditional generation on urban metrics

---

## References

### Key Papers

1. **"Why Diffusion Models Don't Memorize"** (Bonnaire et al., 2025)
   - Implicit dynamical regularization in diffusion training
   - k-NN gap ratio for memorization detection
   - Discrete-time DDPM framework

2. **"Improved Denoising Diffusion Probabilistic Models"** (Nichol & Dhariwal, 2021)
   - Cosine schedule for improved signal preservation
   - Discrete timesteps with learned variance

3. **"Score-Based Generative Modeling through SDEs"** (Song et al., 2021)
   - Continuous-time formulation
   - VP and VE SDE formulations
   - Theoretical foundation for EDM

4. **"Elucidating the Design Space of Diffusion-Based Generative Models"** (Karras et al., 2022)
   - EDM formulation with improved sampling
   - Continuous noise levels
   - State-of-the-art performance

5. **"Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering"** (Wang et al., 2024)
   - Low-dimensional manifold learning
   - Subspace clustering in diffusion models
   - Theoretical analysis of intrinsic dimensions

### Code Repositories

1. **Why-Diffusion-Models-Don-t-Memorize-main**:
   - Discrete-time DDPM implementation
   - Memorization tracking and evaluation
   - Fixed-time training mode

2. **Diffusion-Model-Generalizability-main / edm**:
   - NVIDIA's EDM implementation (continuous-time)
   - Jacobian rank analysis
   - SSCD reproducibility metrics

3. **ddpm-win-master**:
   - Our discrete-time implementation
   - Config-based training with dual GPU support
   - Foundation for urban vitality research

---

## Conclusion

This research uses **discrete-timestep diffusion models** to study how neural networks learn and represent urban vitality. The choice of discrete over continuous-time formulations is driven by:

1. **Research goal alignment**: Discrete milestones naturally answer "when does memorization occur?"
2. **Methodological fit**: Logarithmic checkpointing maps to training iterations
3. **Pragmatic considerations**: Simpler implementation, sufficient for theoretical validation
4. **Schedule flexibility**: Can implement VP/VE schedules within discrete framework

The core hypothesis is that diffusion models learn low-dimensional structural concepts of urban vitality before memorizing idiosyncratic noise. By tracking memorization dynamics across different noise schedules, we aim to identify the "generalization window"—the training phase where structural understanding is maximized while memorization is minimized.

This approach positions diffusion models not as tools for automated design, but as **epistemic probes** that reveal the hidden generative structure of urban life.
