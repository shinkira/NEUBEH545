# NEUBEH 545 – Quantitative Methods in Neuroscience
### MATLAB Tutorials

This repository contains the MATLAB tutorial scripts for **NEUBEH/PBIO 545: Quantitative Methods in Neuroscience**, a graduate course at the University of Washington. The tutorials provide hands-on, interactive introductions to the mathematical and computational tools central to modern systems neuroscience. Each script is designed to be run section by section in MATLAB, with exercises and homework problems embedded throughout.

---

## Tutorials

### 1. Linear Algebra — `LinearAlgebra.m`
*Author: Fred Rieke*

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
*Author: Fred Rieke*

A self-contained introduction to PCA grounded in real electrophysiology data. The tutorial builds from first principles — covariance, correlation, and the eigensystem — to a practical application characterizing variability in rod photoreceptor single-photon responses. Topics include:

- **Motivation for dimensionality reduction**: the bias-variance tradeoff in neural data analysis
- **Covariance and the covariance matrix**: construction, interpretation, and the correlation matrix
- **Eigensystem of the covariance matrix**: how eigenvectors define natural axes of variation
- **Spherizing data**: removing correlation using the inverse square root of the covariance matrix
- **Application to rod single-photon responses**: extracting principal components of response variability, comparing singles vs. failures, building a generative model

*Data files required:* `rgc-spike-response.mat`, `RodData.mat`

---

### 3. Fourier Analysis — `FourierTutorial.m`
*Authors: Mike Shadlen, Adrienne Fairhall*

A thorough introduction to the Fourier transform and its applications in neuroscience. The tutorial emphasizes both conceptual understanding and practical fluency with MATLAB's `fft` function. Topics include:

- **Discrete Fourier transform**: time vs. frequency representation, sampling rate, Nyquist limit
- **Properties of the Fourier transform**: linearity, time-shifting, symmetry of real signals
- **Spectral analysis**: power spectra, amplitude and phase spectra
- **Convolution**: definition, the convolution theorem, and why it matters in neural systems
- **Filtering**: linear time-invariant systems, impulse responses, and filtering in the frequency domain
- **Applications**: analyzing neural signals, understanding sensory system linear filters

---

### 4. Dynamical Systems & Differential Equations — `DiffEQTutorial.m`
*Author: Fred Rieke*

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
*Authors: Michael N. Shadlen, Greg Horwitz*

An introduction to stochastic processes as they apply to neural spike trains. The tutorial begins with renewal processes and the statistical description of interspike intervals (ISIs), then builds toward a rigorous treatment of the Poisson process and its remarkable properties. Topics covered include:

- **Renewal processes** and the iid interval assumption
- **Descriptive statistics** of spike trains: mean, variance, coefficient of variation (CV), and Fano factor
- **The Poisson process**: exponential ISI distribution, memorylessness, the hazard function
- **Gamma processes**: integrate-and-fire intuition, relationship between CV and regularity
- **Entropy** as a measure of spike train disorderliness
- **Markov chains**: transition matrices, steady-state distributions, eigensystem analysis, and the gambler's ruin

*Dependencies:* `plot1ras.m`, `plot2ras.m`

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

1. Clone or download this repository.
2. Add the repository folder to your MATLAB path: `addpath('/path/to/NEUBEH545')`.
3. Open a tutorial script (e.g., `LinearAlgebra.m`) and run it section by section using **Ctrl+Enter** (or **Cmd+Enter** on Mac) to execute one cell at a time.
4. Read the comments carefully — the narrative explanations and homework questions are embedded in the code.

MATLAB R2014b or later is recommended. No additional toolboxes beyond the Statistics and Signal Processing Toolboxes are required.

---

## Course Information

**Course:** NEUBEH/PBIO 545 – Quantitative Methods in Neuroscience  
**Institution:** University of Washington  
**Topics:** Linear algebra, dimensionality reduction, Fourier analysis, dynamical systems, stochastic processes  
