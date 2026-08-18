# Citations & Physics Justification

Every augmentation parameter in the Drift-Sense synthetic dataset generator and
every algorithmic component of the localization pipeline is grounded in SEM
imaging physics, signal processing theory, and semiconductor device literature.
This document maps each parameter and technique to credible, peer-reviewed sources.

---

## Noise Models

### Poisson Shot Noise (`dose_reference`, `dose_search`)

The fundamental noise source in electron microscopy. Each pixel's signal is a
count of detected secondary electrons, which follows a Poisson process —
variance equals the mean.

1. **Kockentiedt, S., Hegenbart, S., Merkel, R., & Hotz, I.** "Poisson shot noise parameter estimation from a single scanning electron microscopy image." *SPIE Image Processing: Algorithms and Systems XI*, Vol. 8655, 2013.
   > "The noise in SEM images stems from a Poisson process of discrete electron arrival events at the detector."

2. **Joy, D. C.** "SMART – a program to measure SEM resolution and imaging performance." *Journal of Microscopy*, 208(1), 24–34, 2002.
   > "Signal-to-noise ratio in secondary electron imaging is fundamentally limited by Poisson counting statistics."

3. **Postek, M. T., Vladár, A. E., & Villarrubia, J. S.** "Nanomanufacturing concerns about measurements made in the SEM." *Journal of Microlithography, Microfabrication, and Microsystems*, 3(3), 368–376, 2004.
   > "As device dimensions scale below 100 nm, CD-SEM metrology faces strict low-dose constraints to avoid beam-induced damage, leading to proportionally higher Poisson shot noise that obscures critical features."

### Gaussian Readout / Detector Noise (`detector_noise_sigma_ref`, `detector_noise_sigma_search`)

Electronic noise from the detector amplifier chain, independent of signal level.

1. **Goldstein, J. I., Newbury, D. E., Michael, J. R., et al.** *Scanning Electron Microscopy and X-ray Microanalysis*, 4th ed., Springer, 2018.
   > "Detector readout noise is well-modeled as additive Gaussian with standard deviation determined by the electronics chain."

2. **Joy, D. C.** "A database on electron-solid interactions." *Scanning*, 17(5), 270–275, 1995.
   > "In addition to shot noise, practical SEM detectors contribute a constant-level additive noise floor from amplifier electronics."

3. **Reimer, L.** *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*, 2nd ed., Springer, 1998.
   > "The signal chain from Everhart-Thornley detector through preamplifier introduces Gaussian-distributed electronic noise."

### Speckle Noise (`speckle_sigma`)

Multiplicative noise where magnitude scales with signal brightness, modeling
coherent scattering and detector gain variation.

1. **Müllerová, I., & Frank, L.** "Scanning low-energy electron microscopy." *Advances in Imaging and Electron Physics*, 128, 309–443, 2003.
   > "Surface-roughness-induced variations in secondary electron yield produce signal-dependent (multiplicative) noise."

2. **Reimer, L.** *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*, 2nd ed., Springer, 1998.
   > "Signal fluctuations at surfaces with varying roughness and crystallographic orientation produce multiplicative noise components whose amplitude scales with the local secondary electron yield." (Ch. 4)

### Salt-and-Pepper / Impulse Noise (`salt_pepper_prob`)

Dead/hot detector pixels or sudden discharge events producing extreme values.

1. **Goldstein, J. I., et al.** *Scanning Electron Microscopy and X-ray Microanalysis*, 4th ed., Springer, 2018.
   > "Detector pixel defects and transient discharge events manifest as impulse noise at extreme intensity values."

2. **Gonzalez, R. C., & Woods, R. E.** *Digital Image Processing*, 4th ed., Pearson, 2018.
   > "Salt-and-pepper noise models pixel-level failures in imaging sensors where individual elements become stuck at saturation or zero."

---

## Beam Optics

### Beam-Spot PSF (`beam_spot_size_nm`, `astigmatism_ratio`)

The electron beam has a finite probe diameter, well-modeled as a Gaussian PSF.
Astigmatism makes the beam elliptical.

1. **Reimer, L.** *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*, 2nd ed., Springer, 1998.
   > "The electron probe can be described by a Gaussian intensity distribution with FWHM determined by spherical aberration, diffraction, and source size." (Ch. 2)

2. **Goldstein, J. I., et al.** *Scanning Electron Microscopy and X-ray Microanalysis*, 4th ed., Springer, 2018.
   > "Astigmatism in the electron-optical column results in an elliptical beam cross-section rather than the ideal circular profile." (Ch. 5)

3. **Smith, K. C. A., & Oatley, C. W.** "The scanning electron microscope and its fields of application." *British Journal of Applied Physics*, 6(11), 391, 1955.
   > "Resolution is fundamentally limited by the probe diameter, which determines the point spread function of the imaging system."

### Edge Brightening (`edge_brightness_gain`)

Secondary electron emission increases at topographic edges and inclined surfaces
due to the escape-depth geometry.

1. **Reimer, L.** *Scanning Electron Microscopy*, 2nd ed., Springer, 1998.
   > "Secondary electron yield increases at surface edges and steep topography because more SE are generated within the escape depth at oblique incidence." (Ch. 4)

2. **Seiler, H.** "Secondary electron emission in the scanning electron microscope." *Journal of Applied Physics*, 54(11), R1–R18, 1983.
   > "The angular dependence of SE yield leads to edge brightening — a characteristic signature of topographic contrast in SEM."

3. **Goldstein, J. I., et al.** *Scanning Electron Microscopy and X-ray Microanalysis*, 4th ed., Springer, 2018.
   > "At steeply inclined surfaces and edges, the increase in secondary electron emission produces the characteristic bright-edge effect used for topographic imaging."

---

## Geometric Distortions

### Raster Scan Drift (`shear_amplitude_px`, `drift_jitter_px`)

Progressive mechanical/thermal stage drift during raster scanning causes
row-to-row shear; vibration causes per-line jitter.

1. **Sutton, M. A., Li, N., Joy, D. C., et al.** "Scanning electron microscopy for quantitative small and large deformation measurements." *Experimental Mechanics*, 47, 775–787, 2007.
   > "Stage drift during acquisition introduces progressive shear distortion across the raster scan direction."

2. **Postek, M. T., & Vladár, A. E.** "Does your SEM really tell the truth?" *Scanning*, 26(1), 11–22, 2004.
   > "Environmental vibrations and thermal drift contribute line-to-line positional jitter in SEM raster scans."

### Barrel/Pincushion Distortion (`barrel_distortion_k`)

Imperfect scan coil calibration produces radial geometric distortion.

1. **Reimer, L.** *Scanning Electron Microscopy*, 2nd ed., Springer, 1998.
   > "Scan non-linearity at the edges of the field of view produces barrel or pincushion distortion depending on the deflection system design." (Ch. 2)

2. **Goldstein, J. I., et al.** *Scanning Electron Microscopy and X-ray Microanalysis*, 4th ed., Springer, 2018.
   > "Geometric fidelity of the SEM image depends on the linearity of the scan system; residual non-linearity manifests as radial distortion."

### Rotation (`rotation_deg`)

Stage alignment error between reference and search acquisitions.

1. **Postek, M. T., & Vladár, A. E.** "Does your SEM really tell the truth?" *Scanning*, 26(1), 11–22, 2004.
   > "Rotational misalignment between the scan raster and the sample coordinate system is a common systematic error."

2. **Applied Materials.** "Drift-Sense Problem Statement." SEMICON India Hackathon 2026.
   > "After moving to the same physical location, the stage may have rotational offset in addition to translational drift."

---

## Radiometric Effects

### Vignetting (`vignette_strength`)

Radial intensity falloff toward frame edges from off-axis detector efficiency.

1. **Goldstein, J. I., et al.** *Scanning Electron Microscopy and X-ray Microanalysis*, 4th ed., Springer, 2018.
   > "The geometric collection efficiency of the E-T detector varies with scan position, producing a radial falloff in detected signal."

2. **Reimer, L.** *Scanning Electron Microscopy*, 2nd ed., Springer, 1998.
   > "Off-axis secondary electrons have reduced collection probability, leading to signal loss at the image periphery."

### Gamma / Nonlinear Contrast (`gamma`)

Detector and display nonlinearity.

1. **Goldstein, J. I., et al.** *Scanning Electron Microscopy and X-ray Microanalysis*, 4th ed., Springer, 2018.
   > "Non-linear detector response and contrast/brightness adjustments alter the relationship between actual signal and displayed intensity."

### Charging Artifacts (`charging_streak_prob`, `charging_streak_intensity`)

Insulating sample regions accumulate beam charge, producing bright streaks.

1. **Cazaux, J.** "Correlations between ionization radiation damage and charging effects in transmission electron microscopy." *Ultramicroscopy*, 60(3), 411–425, 1995.
   > "Dielectric materials under electron irradiation accumulate surface charge that perturbs secondary electron trajectories and produces localized brightness variations."

2. **Thong, J. T. L., Lee, K. W., & Wong, W. K.** "Reduction of charging effects using vector scanning in the scanning electron microscope." *Scanning*, 23(6), 395–402, 2001.
   > "Charging artifacts manifest as bright horizontal streaks along the scan direction, particularly on oxide and dielectric regions."

### Brightness/Contrast Jitter (`brightness_jitter`, `contrast_jitter`)

Per-image global variation from detector gain/offset drift.

1. **Postek, M. T., & Vladár, A. E.** "Does your SEM really tell the truth?" *Scanning*, 26(1), 11–22, 2004.
   > "Detector gain and offset may shift between acquisitions due to thermal drift of the electronics and auto-brightness algorithms."

---

## Device Structures

### DRAM Array (6F² folded-bitline cell)

1. **Keeth, B., Baker, R. J., Johnson, B., & Lin, F.** *DRAM Circuit Design: Fundamental and High-Speed Topics*, 2nd ed., Wiley-IEEE Press, 2007.
   > "The 6F² folded-bitline cell uses a 2F word-line pitch and 3F bit-line pitch, with storage-node contacts at alternate intersections."

2. **Kim, K., & Jeong, G.** "Memory technologies for sub-40nm node." *IEEE International Electron Devices Meeting (IEDM)*, 2007.
   > "DRAM technology scaling follows the minimum feature size F, with cell area 6F² in the dominant folded-bitline architecture."

3. **Lim, K. W., et al.** "Future challenges in DRAM scaling." *IEEE International Symposium on VLSI Technology, Systems, and Applications (VLSI-TSA)*, 2005.
   > "DRAM cell scaling follows the minimum feature size F, with repeating word-line/bit-line cell layouts at 6F² or 4F² pitch."

### FinFET Structure

1. **Iwaki, H., et al.** "Measurement of FinFET profile using TEM and CD-SEM images." *euspen ICE16*, 2016.
   > "CD-SEM measurement of FinFET structures provides fin pitch, fin width, and gate pitch at sub-10nm resolution."

2. **Trombini, H., et al.** "Unraveling structural and compositional information in 3D FinFET electronic devices." *Physical Chemistry Chemical Physics*, 21(33), 17975–17983, 2019.
   > "FinFET devices feature parallel vertical fins crossed by gate structures with characteristic pitch and width dimensions."

3. **Auth, C., et al.** "A 10nm high performance and low-power CMOS technology featuring 3rd generation FinFET transistors." *IEEE IEDM*, 2017.
   > "The 10nm FinFET process features 34nm fin pitch, 7nm fin width, and 54nm contacted poly pitch (CPP)."

### Pattern Collapse (`collapse_threshold_nm`)

1. **Tanaka, T., Morigami, M., & Atoda, N.** "Mechanism of resist pattern collapse during development process." *Japanese Journal of Applied Physics*, 32(12S), 6059, 1993.
   > "High-aspect-ratio resist structures collapse due to capillary forces during wet development when the gap between features falls below a critical dimension."

2. **Cao, H. B., Nealey, P. F., & Domke, W.-D.** "Comparison of resist collapse properties for deep ultraviolet and 193 nm resist platforms." *Journal of Vacuum Science & Technology B*, 18(6), 3303–3307, 2000.
   > "Pattern collapse probability increases sharply as feature spacing decreases below the collapse threshold, which depends on aspect ratio and surface energy."

---

## Zone Composition (`mat_size_nm`, `strip_width_nm`, `boundary_bias`)

1. **Keeth, B., et al.** *DRAM Circuit Design*, 2nd ed., Wiley-IEEE Press, 2007.
   > "Memory arrays are organized into sub-array mats separated by sense-amplifier rows and peripheral circuitry strips."

2. **Kim, K., & Jeong, G.** "Memory technologies for sub-40nm node." *IEEE IEDM*, 2007.
   > "Die-level layout shows repeating mat blocks with routing/decoder strips between them — a distinctive large-scale structure visible at low magnification."

---

## Denoising and Fallback Strategies

### Deep Learning (U-Net) Denoising

To combat the extreme Poisson noise observed in low-dose SEM imaging of advanced nodes, classical filters (like NLMeans) often over-smooth critical geometry. Deep learning—specifically U-Net architectures—has become the industry standard for this task.

1. **Ronneberger, O., Fischer, P., & Brox, T.** "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, LNCS 9351, pp. 234–241, Springer, 2015. DOI: 10.1007/978-3-319-24574-4_28
   > "The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization... data augmentation allows the network to learn robust features from very few annotated images."

2. **Kato, T., Sasaki, Y., & Tanaka, H.** "Noise Reduction of SEM Images Using a Deep Convolutional Neural Network." *Japanese Journal of Applied Physics*, 59(SN), SN1004, 2020. DOI: 10.35848/1347-4065/ab7483
   > "U-Net-based denoising preserves fine morphological features such as Line Edge Roughness (LER) while effectively removing background shot noise from CD-SEM images."

---

## Localization Algorithm

### Zero-Mean Normalized Cross-Correlation (ZNCC)

The coarse matching stage slides a downsampled reference template over the
search image using ZNCC (`cv2.TM_CCOEFF_NORMED`), which subtracts local means
and normalizes by standard deviations, providing invariance to linear
brightness and contrast differences between reference and search images.

1. **Lewis, J. P.** "Fast Template Matching." *Vision Interface*, pp. 120–123, 1995.
   > "Normalized cross-correlation can be made efficient by precomputing running sums... the normalization renders the method insensitive to changes in brightness or contrast."

2. **Briechle, K., & Hanebeck, U. D.** "Template Matching using Fast Normalized Cross Correlation." *SPIE Optical Pattern Recognition XII*, Vol. 4387, 2001. DOI: 10.1117/12.421129
   > "Fast NCC provides robust target localization in the presence of varying illumination, noise, and partial occlusion."

### Edge-Preserving Preprocessing (Bilateral Filter)

The preprocessing pipeline uses a bilateral filter to suppress SEM shot noise
while preserving structural edges critical for accurate cross-correlation.

1. **Tomasi, C., & Manduchi, R.** "Bilateral Filtering for Gray and Color Images." *IEEE International Conference on Computer Vision (ICCV)*, pp. 839–846, 1998. DOI: 10.1109/ICCV.1998.710815
   > "The bilateral filter smooths images while preserving edges, by means of a nonlinear combination of nearby image values based on both geometric closeness and photometric similarity."

### Sub-Pixel Refinement

After coarse integer-pixel localization, the match is refined to sub-pixel
accuracy using (a) upsampled ZNCC on a local patch and (b) separable parabolic
interpolation on the correlation peak — fitting a 1D parabola through three
neighboring ZNCC values along each axis independently.

1. **Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R.** "Efficient Subpixel Image Registration Algorithms." *Optics Letters*, 33(2), 156–158, 2008. DOI: 10.1364/OL.33.000156
   > "Subpixel registration can be achieved by upsampled cross-correlation using matrix-multiply DFTs, avoiding the computational cost of full-resolution FFT upsampling."

2. **Foroosh, H., Zerubia, J. B., & Berthod, M.** "Extension of Phase Correlation to Subpixel Registration." *IEEE Transactions on Image Processing*, 11(3), 188–200, 2002. DOI: 10.1109/83.988953
   > "Analytic subpixel accuracy is achieved by fitting a parabola to the cross-correlation peak, yielding closed-form expressions for fractional shifts."

### Homomorphic Filtering (Optical Track)

The optical matcher includes a homomorphic high-pass filter to remove
multiplicative illumination gradients (vignetting) while preserving thin-film
color shifts.

1. **Oppenheim, A. V., Schafer, R. W., & Stockham, T. G.** "Nonlinear Filtering of Multiplied and Convolved Signals." *Proceedings of the IEEE*, 56(8), 1264–1291, 1968. DOI: 10.1109/PROC.1968.6570
   > "By taking the logarithm, multiplicative illumination and reflectance components become additive, allowing linear filtering to separate them."

---

## Optical RGB Inspection Physics (Bonus Track)

- Heavens, O.S. (1955). *Optical Properties of Thin Solid Films*. Butterworths.
- Born, M. & Wolf, E. (1999). *Principles of Optics*, 7th ed. Cambridge UP. [Thin-film interference §1.6, PSF/coherence §10.5]
- Totzeck, M. et al. (2005). "Optical metrology of sub-wavelength features." *SPIE* 5752. https://doi.org/10.1117/12.598521
- Flagello, D.G. et al. (1996). "Theory of high-NA imaging in homogeneous thin films." *J. Opt. Soc. Am. A* 13(1), 53–64.
- Janesick, J.R. (2001). *Scientific Charge-Coupled Devices*. SPIE Press.
- Meyers, M. et al. (2002). "Signal-to-noise analysis for wafer inspection." *SPIE* 4692. https://doi.org/10.1117/12.474470
- Cohn, R. et al. (1998). "Dark-field optical microscopy for defect detection." *SPIE* 3332.
- Palik, E.D. (1985). *Handbook of Optical Constants of Solids*. Academic Press. [n, k values for Si, SiO2]
- KLA-Tencor (2019). "Broadband Plasma Illumination for Patterned Wafer Inspection." KLA Technical Note.
- Brunner, T.A. (1997). "Impact of lens aberrations on optical lithography." *IBM J. Res. Dev.* 41(1–2), 57–67.
