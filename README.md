# NEUBEH 545 – Quantitative Methods in Neuroscience
### MATLAB and Python Tutorials

This repository contains the tutorial material for **NEUBEH/PBIO 545: Quantitative Methods in Neuroscience**, a graduate course at the University of Washington. The tutorials provide hands-on, interactive introductions to the mathematical and computational tools central to modern systems neuroscience. Each one is designed to be worked through section by section, with the narrative explanations and homework problems embedded alongside the code.


## Repository Layout

```
matlab/     MATLAB tutorial scripts, helper functions, and data files
python/     Jupyter notebook versions of the tutorials
```

Every tutorial is available in both languages. The MATLAB scripts are the
original course material; the Python notebooks cover the same seven topics,
with the narrative in markdown cells and each core method implemented from
scratch before being checked against the SciPy/scikit-learn equivalent. See
`python/README.md` for setup.

---

## Tutorials

File paths below are relative to `matlab/`.

### 1. Linear Algebra — `LinearAlgebra.m`
*Also available as:* [`python/LinearAlgebra.ipynb`](python/LinearAlgebra.ipynb)

A visual and applied introduction to linear algebra, progressing from vector geometry through matrix transformations to the eigensystem. The tutorial emphasizes geometric intuition alongside computation. Topics include:

- **Vectors and vector spaces**: representation, addition, scalar multiplication
- **Matrix operations**: matrix multiplication, the inverse, solving systems of linear equations
- **Geometric transformations**: scaling, rotation, projection
- **The column space** and null space (see also `ColumnSpaceDemo.m`)
- **Eigenvectors and eigenvalues**: definition, computation, and geometric meaning
- **Matrix powers and the Fibonacci sequence**: connecting eigensystems to dynamical systems
- **Singular value decomposition (SVD)**: introduction and applications

*Dependencies:* `PlotVector.m`, `Plot3DVector.m`

---

### 2. Principal Components Analysis — `PCATutorial.m`
*Also available as:* [`python/PCATutorial.ipynb`](python/PCATutorial.ipynb)

A self-contained introduction to PCA grounded in real electrophysiology data. The tutorial builds from first principles — covariance, correlation, and the eigensystem — to a practical application characterizing variability in rod photoreceptor single-photon responses. Topics include:

- **Motivation for dimensionality reduction**: the bias-variance tradeoff in neural data analysis
- **Covariance and the covariance matrix**: construction, interpretation, and the correlation matrix
- **Eigensystem of the covariance matrix**: how eigenvectors define natural axes of variation
- **Spherizing data**: removing correlation using the inverse square root of the covariance matrix
- **Application to rod single-photon responses**: extracting principal components of response variability, comparing singles vs. failures, building a generative model

*Data files required:* `rgc-spike-response.mat`, `RodData.mat`

---

### 3. Fourier Analysis — `FourierTutorial.m`
*Also available as:* [`python/FourierTutorial.ipynb`](python/FourierTutorial.ipynb)

A thorough introduction to the Fourier transform and its applications in neuroscience. The tutorial emphasizes both conceptual understanding and practical fluency with MATLAB's `fft` function. Topics include:

- **Discrete Fourier transform**: time vs. frequency representation, sampling rate, Nyquist limit
- **Properties of the Fourier transform**: linearity, time-shifting, symmetry of real signals
- **Spectral analysis**: power spectra, amplitude and phase spectra
- **Convolution**: definition, the convolution theorem, and why it matters in neural systems
- **Filtering**: linear time-invariant systems, impulse responses, and filtering in the frequency domain
- **Applications**: analyzing neural signals, understanding sensory system linear filters

---

### 4. Dynamical Systems & Differential Equations — `DiffEQTutorial.m`
*Also available as:* [`python/DiffEQTutorial.ipynb`](python/DiffEQTutorial.ipynb)

A practical introduction to ordinary differential equations (ODEs) solved numerically using the finite difference (Euler) method. The tutorial is organized around a biologically grounded sequence of examples culminating in a mechanistic model of phototransduction. Topics include:

- **Linear first-order ODEs**: numerical vs. analytical solutions, role of initial conditions
- **Spontaneous activation and feedback**: nonlinear ODE systems, gain and cooperativity
- **Phototransduction model**: linked differential equations for rhodopsin, phosphodiesterase, cGMP, and membrane current
- **Calcium feedback**: Hill equation, negative feedback dynamics, damped oscillations
- **Hodgkin-Huxley gating particles**: voltage-dependent alpha/beta rate constants, the m-gate
- **Fourier methods for ODEs**: solving differential equations in the frequency domain

*Dependencies:* `dydt.m`, `dydt10.m`

---

### 5. Stochastic Processes — `stochasticProcessesTutorial.m`
*Also available as:* [`python/StochasticProcessesTutorial.ipynb`](python/StochasticProcessesTutorial.ipynb)

An introduction to stochastic processes as they apply to neural spike trains. The tutorial begins with renewal processes and the statistical description of interspike intervals (ISIs), then builds toward a rigorous treatment of the Poisson process and its remarkable properties. Topics covered include:

- **Renewal processes** and the iid interval assumption
- **Descriptive statistics** of spike trains: mean, variance, coefficient of variation (CV), and Fano factor
- **The Poisson process**: exponential ISI distribution, memorylessness, the hazard function
- **Gamma processes**: integrate-and-fire intuition, relationship between CV and regularity
- **Entropy** as a measure of spike train disorderliness
- **Markov chains**: transition matrices, steady-state distributions, eigensystem analysis, and the gambler's ruin

*Dependencies:* `plot1ras.m`, `plot2ras.m`

---

## Population Analysis Tutorials

These two tutorials extend the core material to simulated neural population
recordings. Both are self-contained: the population is simulated inside the
script, so no data files are needed, and all helper functions are defined at
the bottom of each file.

### 6. PCA on Neural Populations — `PCANeuroPopTutorial.m`

A companion to `PCATutorial.m` that applies PCA to a simulated population of orientation-tuned visual neurons, shifting the emphasis from variability in single-cell responses to the geometry of population activity. Topics include:

- **Why populations live in low-dimensional subspaces**: two-neuron intuition, shared vs. private input
- **Covariance and its eigensystem**: built by hand, then verified against MATLAB's `pca`
- **The orientation ring**: how a one-dimensional stimulus variable becomes a closed curve in PC space
- **PC loadings as tuning-curve structure**: cosine and sine profiles over preferred orientation
- **Decoding from the manifold**: recovering stimulus orientation with `atan2` on the first two PCs
- **Tuning width and dimensionality**: how sharpening tuning changes the variance spectrum
- **Two-dimensional stimuli**: orientation × spatial frequency and the toroidal manifold
- **Fixed-axis projection**: comparing two populations in a common PC basis to quantify manifold deformation

*Prerequisites:* `LinearAlgebra.m`, `PCATutorial.m`  
*Also available as:* [`python/PCANeuroPopTutorial.ipynb`](python/PCANeuroPopTutorial.ipynb)

---

### 7. Pattern Discrimination & Classification — `ClassificationTutorial.m`

An introduction to decoding and classification, built around a fine orientation discrimination task (88° vs. 92°) performed by a simulated population with realistic trial-to-trial variability. The tutorial makes the case that variance and discriminability are different things: the direction that best separates two stimuli is generally not the direction of largest variance. Topics include:

- **Signal detection theory**: criterion, hit and false-alarm rates, ROC curves constructed by hand
- **AUC and d-prime**: the equivalence of AUC and two-alternative forced choice, and when the Gaussian shortcut fails
- **Neurometric functions**: why information lives in tuning-curve *slope* rather than peak response
- **Linear discriminant analysis**: Fisher's criterion, shrinkage regularization, ill-conditioned covariance
- **Logistic regression as a GLM**: fitting by iteratively reweighted least squares, calibration, regularization paths
- **Support vector machines**: margins, support vectors, soft-margin trade-offs, and the RBF kernel
- **Cross-validation**: learning curves, overfitting, nested model selection, permutation tests
- **Decoding geometry**: signal vs. noise correlations and information-limiting correlations
- **Nonlinear classifiers**: k-nearest neighbours and the curse of dimensionality

*Prerequisites:* `LinearAlgebra.m`, `PCANeuroPopTutorial.m`, `stochasticProcessesTutorial.m`  
*Also available as:* [`python/ClassificationTutorial.ipynb`](python/ClassificationTutorial.ipynb)

### 8. Generalized Linear Models — `python/GLMTutorial.ipynb`
*Also available as:* MATLAB version pending

An introduction to GLMs as **encoding** models, complementing the decoding direction taken by
the classification tutorial. Built around a simulated delayed match-to-sample task in a
virtual T-maze, with running speed deliberately correlated with cue identity, so that the
central methodological problem is unavoidable: a neuron driven purely by running speed looks
strongly cue-selective to any analysis that ignores movement. Topics include:

- **The Poisson GLM**: log link, multiplicative effects, likelihood and deviance
- **Fitting by IRLS**: implemented from scratch, then checked against statsmodels
- **Design matrices from trial structure**: cue × epoch indicators, raised-cosine speed bases, time lags
- **Fraction of deviance explained** on held-out data as the GLM analogue of R²
- **Cross-validation by trial rather than frame**, and an honest test of when that matters
- **Elastic-net regularization**: why glmnet uses α ≈ 0.95, and `lambda_min` vs `lambda_1se`
- **Nested model comparison**: ΔFDE as the selectivity measure that survives a movement confound
- **Permutation testing** by shuffling trial labels, and why frame-wise shuffling inflates significance

*Prerequisites:* `LinearAlgebra.m`, `ClassificationTutorial.m`  
*Adapted from:* a DMTS GLM pipeline (`GlmnetDmtsDemo`)

---

---

## Helper Scripts

| File | Description |
|------|-------------|
| `PlotVector.m` | Plots a 2D vector from the origin |
| `Plot3DVector.m` | Plots a 3D vector from the origin |
| `ColumnSpaceDemo.m` | Short demonstration of the column space of a matrix |
| `plot1ras.m` | Plots a single-trial spike raster (by Mike Shadlen) |
| `plot2ras.m` | Plots multi-condition spike rasters (by Greg Horwitz) |
| `dydt.m` / `dydt10.m` | ODE helper functions for differential equation examples |
| `expAsLimit.m` | Demonstrates the exponential function as a limit |
| `threshold.m` | Simple threshold nonlinearity function |

## Data Files

| File | Description |
|------|-------------|
| `rgc-spike-response.mat` | Spike responses from a retinal ganglion cell to a dim flash of light (used in PCATutorial) |
| `RodData.mat` | Single-photon responses from rod photoreceptors, including both "singles" and "failures" trials (used in PCATutorial) |

---

## Getting Started

### MATLAB

1. Clone or download this repository.
2. Add the MATLAB folder to your path: `addpath('/path/to/NEUBEH545/matlab')`.
3. Open a tutorial script (e.g., `LinearAlgebra.m`) and run it section by section using **Ctrl+Enter** (or **Cmd+Enter** on Mac) to execute one cell at a time.
4. Read the comments carefully — the narrative explanations and homework questions are embedded in the code.

MATLAB R2014b or later is recommended. No additional toolboxes beyond the Statistics and Signal Processing Toolboxes are required. `ClassificationTutorial.m` implements its methods from scratch and uses the Statistics Toolbox only for the optional support vector machine section, falling back to a hand-written classifier when `fitcsvm` is unavailable.

### Python / Jupyter

Targets **Python 3.14** (current stable); 3.11 or newer works. See
[`python/README.md`](python/README.md) for full setup. In short:

```bash
cd python
python3.14 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Contributors should run `nbstripout --install` once inside the repository so that notebook outputs stay out of the commit history.

---

## Course Information

**Course:** NEUBEH/PBIO 545 – Quantitative Methods in Neuroscience  
**Institution:** University of Washington  
**Topics:** Linear algebra, dimensionality reduction, Fourier analysis, dynamical systems, stochastic processes, population geometry, neural decoding and classification  
