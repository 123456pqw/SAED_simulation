# SAED_simulation Data

SAED_simulation/
├── run_saed.py          # Main script for SAED simulation
├── file_id.csv          # Mapping of material IDs to material information
└── README.md            # This file

# Electron Diffraction Simulation (MULTEM-based)

This repository provides a multislice-based electron diffraction (ED/CBED/SAED) simulation pipeline built on MULTEM, enabling physically realistic diffraction pattern generation for deep learning and crystallographic analysis.

# Simulation Method

The electron scattering process is simulated using the multislice method with the MULTEM engine, explicitly accounting for dynamical scattering effects. Frozen phonon configurations are used to model thermal vibrations.

# Key physics models include:

Multislice propagation
Frozen phonon approximation
Coherent illumination
3D potential slicing
# Key Physical Parameters
Electron beam settings
Acceleration voltage (energy): 200 keV
Convergence angle (beam semi-angle): 0.6 mrad
Beam type: Coherent illumination
Beam position: Centered at specimen mid-plane

These parameters define the probe size in reciprocal space, i.e., the effective beam convergence.

Beam / probe characteristics
Beam convergence angle controls probe angular spread
Smaller convergence → sharper diffraction spots (SAED-like)
Larger convergence → broader intensity distribution (CBED-like)

Effective beam size is implicitly determined by:

electron wavelength
convergence semi-angle
real-space sampling grid
Detector / collection geometry

A multi-angle detection scheme is used:

collection_angle = [
    ("angle1", 5),
    ("angle2", 10),
    ("angle3", 20),
    ("angle4", 30)
]
Defines angular integration windows in reciprocal space
Used to simulate different experimental detector apertures
Maximum collection angle determines k-space field of view
Thickness sampling

To model realistic specimen effects:

Through-thickness sampling is used
Thickness is discretized as:
thick = [x * spec_dz for x in range(0, 10)]

This allows:

Multiple scattering accumulation
Thickness-dependent intensity evolution
Frozen phonon model (thermal vibrations)

Thermal disorder is included using frozen phonon averaging:

Number of configurations: nphonon = 20
Random seed: pn_seed
Independent atomic displacements per configuration

This ensures:

Thermal diffuse scattering
More realistic intensity distribution
Source and detector broadening

To mimic experimental imperfections:

Defocus spread (temporal incoherence)
Source size broadening (spatial incoherence)

Implemented via Gaussian approximations:

temporal spread σ derived from iehwgd
spatial spread σ derived from hwhm

These parameters simulate:

finite electron source size
lens instabilities
instrumental blur
#  Simulation grid and resolution
Real-space grid: 1024 × 1024
Potential slicing: dz_Proj, spec_dz = 2.0 Å
Inner sampling: adaptive from crystal size

Reciprocal-space resolution is determined by:

maximum collection angle
detector pixel size
real-space box size

# Output generation

For each simulation:

Multislice propagation is performed
Multiple frozen phonon configurations are averaged
Intensities are combined:
cbed_avg = np.abs(sum(data)) / len(data)
Log transformation is applied:
log(I + ε)
Final diffraction pattern is mapped to reciprocal space

# Reproducibility

All simulations are fully reproducible by fixing:

CIF structure
zone axis
random seed (phonon configurations)
beam convergence angle
thickness sampling strategy

# Notes
Beam convergence angle acts as a key parameter controlling diffraction sharpness
Frozen phonon averaging significantly improves realism for ML training datasets

# Output
Each simulation produces:

2D diffraction intensity map
Reciprocal-space calibrated image
Optional multi-thickness stacks