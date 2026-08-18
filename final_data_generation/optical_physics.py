import cv2
import numpy as np
from scipy.ndimage import fourier_gaussian


# ---------------------------------------------------------------------------
# 1. Thin-Film Interference (Color Shifting)
# ---------------------------------------------------------------------------
def thin_film_reflectance(thickness_map, wavelengths_nm, n_film, n_sub_complex):
    """
    Simulates constructive/destructive interference in thin dielectric films.
    thickness_map : 2D array in nm (representing oxide thickness variation)
    wavelengths_nm: array [R_wl, G_wl, B_wl], e.g. [650, 550, 450]
    n_film        : refractive index of film (real, scalar or per-wavelength)
    n_sub_complex : complex refractive index of substrate per wavelength
    returns       : (H, W, 3) float array of reflectance per channel
    """
    n_air = 1.0
    H, W = thickness_map.shape
    rgb = np.zeros((H, W, 3))

    for i, (wl, nf, ns) in enumerate(zip(wavelengths_nm, n_film, n_sub_complex)):
        r1 = (n_air - nf) / (n_air + nf)
        r2 = (nf - ns) / (nf + ns)
        delta = (4 * np.pi * nf * thickness_map) / wl   # radians
        phase = np.exp(1j * delta)
        r_total = (r1 + r2 * phase) / (1 + r1 * r2 * phase)
        rgb[:, :, i] = np.abs(r_total)**2

    return rgb


# ---------------------------------------------------------------------------
# 2. Chromatic Aberration & Defocus
# ---------------------------------------------------------------------------
def chromatic_psf_blur(image_rgb, na=0.9, wavelengths_nm=None, defocus_nm=0.0,
                        pixel_size_nm=10.0, chromatic_shift_nm=None):
    """
    Simulates optical diffraction limit and axial chromatic aberration.
    """
    if chromatic_shift_nm is None:
        chromatic_shift_nm = [0, 0, 0]
    if wavelengths_nm is None:
        wavelengths_nm = [650, 550, 450]
    if chromatic_shift_nm is None:
        chromatic_shift_nm = [15, 0, -20]   # typical residual CA in nm

    blurred = np.zeros_like(image_rgb)
    for i, (wl, ca) in enumerate(zip(wavelengths_nm, chromatic_shift_nm)):
        total_defocus = defocus_nm + ca
        # Incoherent diffraction limit (Airy radius in pixels)
        r_airy_px = (0.61 * wl / na) / pixel_size_nm
        # Defocus broadening (geometric radius)
        sigma_defocus_px = abs(total_defocus) * na / pixel_size_nm
        # Quadrature sum (PSF convolution approximated as Gaussian for speed)
        sigma_total = np.sqrt(r_airy_px**2 + sigma_defocus_px**2)
        
        # Apply slight lateral chromatic shift for edge color fringing
        lateral_shift_px = ca / pixel_size_nm
        
        # Blur the channel using Fourier space for extreme performance
        # Spatial Gaussian filter is O(N*K), Fourier is O(N log N).
        # For sigma=440 on a 3000x3000 image, this is ~1000x faster!
        ch_fourier = np.fft.fft2(image_rgb[:, :, i])
        ch_blurred_fourier = fourier_gaussian(ch_fourier, sigma=sigma_total)
        del ch_fourier  # Free 800MB complex array immediately
        blurred_ch = np.fft.ifft2(ch_blurred_fourier).real
        del ch_blurred_fourier  # Free another 800MB
        
        # Shift the channel laterally to simulate transverse chromatic aberration
        M = np.float32([[1, 0, lateral_shift_px], [0, 1, lateral_shift_px]])
        blurred[:, :, i] = cv2.warpAffine(blurred_ch, M, (blurred_ch.shape[1], blurred_ch.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    return blurred


# ---------------------------------------------------------------------------
# 3. Sensor Noise Model
# ---------------------------------------------------------------------------
def apply_sensor_noise(image_photons, qe=0.75, gain_e_per_dn=0.5,
                        read_noise_e=5.0, dark_current_e=0.1,
                        fpn_sigma=0.01, seed=None):
    """
    Simulates CMOS sensor noise stack: QE, fixed-pattern noise, shot noise, dark current, and read noise.
    Returns: (H, W, 3) uint8 simulated sensor output
    """
    rng = np.random.default_rng(seed)
    electrons = image_photons * qe

    # Fixed-pattern noise: pixel-to-pixel QE variation
    fpn = rng.normal(1.0, fpn_sigma, size=electrons.shape)
    electrons = electrons * fpn

    # Shot noise: Poisson on electrons
    shot = rng.poisson(np.clip(electrons, 0, None).astype(float))

    # Dark current + read noise
    dark = rng.poisson(dark_current_e, size=shot.shape)
    read = rng.normal(0, read_noise_e, size=shot.shape)

    total_e = shot + dark + read
    # Optical 8-bit scale rather than 16-bit to match our save format (png)
    # Calibrated gain so max brightness doesn't clip too hard (Boosted by 2.0x for brightness)
    dn = np.clip((total_e / gain_e_per_dn) / 128.0, 0, 255).astype(np.uint8)
    return dn


# ---------------------------------------------------------------------------
# 4. Illumination Gradients
# ---------------------------------------------------------------------------
def illumination_gradient(shape, pixel_size_nm=1.0, mode='brightfield', vignette_sigma=None, seed=None):
    """
    Simulates Köhler illumination falloff (vignetting) and lamp non-uniformity.
    """
    H, W = shape
    rng = np.random.default_rng(seed)
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2 + yy**2

    if mode == 'brightfield':
        # Always generate the noise profile for the full camera sensor (10000nm x 10000nm)
        sensor_size = int(10000 / pixel_size_nm)
        
        # Vignette centered on the sensor
        y_sensor = np.linspace(-1, 1, sensor_size)
        x_sensor = np.linspace(-1, 1, sensor_size)
        xx_s, yy_s = np.meshgrid(x_sensor, y_sensor)
        r2_s = xx_s**2 + yy_s**2
        
        if vignette_sigma is None:
            vignette_sigma = 1.2
        vignette_sensor = np.exp(-r2_s / (2 * vignette_sigma**2))

        # Low-frequency lamp non-uniformity: smooth random field on the sensor
        noise_dim = max(1, sensor_size // 16)
        noise_lowfreq = rng.normal(0, 0.03, size=(noise_dim, noise_dim))
        noise_upsampled = cv2.resize(noise_lowfreq, (sensor_size, sensor_size), interpolation=cv2.INTER_CUBIC)

        illum_sensor = vignette_sensor * (1.0 + noise_upsampled)
        
        # Crop the requested shape from the center of the sensor
        start_y = (sensor_size - H) // 2
        start_x = (sensor_size - W) // 2
        illum = illum_sensor[start_y:start_y+H, start_x:start_x+W]

    elif mode == 'darkfield':
        # Annular ring: peak at mid-radius
        r_peak = 0.6
        sigma_ring = 0.15
        illum = np.exp(-((np.sqrt(r2) - r_peak)**2) / (2 * sigma_ring**2))
        illum = illum / illum.max()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return illum

def apply_illumination(image_rgb, illum_map):
    return image_rgb * illum_map[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Core Wrappers for Generator Pipeline
# ---------------------------------------------------------------------------
def simulate_rgb_wafer_image(height_map_nm, thickness_map_nm,
                              pixel_size_nm=10.0, defocus_nm=0.0,
                              na=0.9, photon_flux=20000, mode='brightfield', seed=None):
    """
    Full optical simulation pipeline.
    height_map_nm: (H, W) array representing physical topography.
    """
    # Base material indices
    wls = [650, 550, 450]
    n_sio2 = [1.457, 1.463, 1.472]
    n_si = [3.85+0.02j, 4.05+0.03j, 4.36+0.08j]

    # Combine topography and film thickness variations
    # Add 600nm of base oxide so the phase wraps multiple times, creating vivid interference colors!
    effective_thickness = height_map_nm + thickness_map_nm + 600.0

    # Step 1: Thin-film color
    rgb = thin_film_reflectance(effective_thickness, wls, n_sio2, n_si)

    # Step 2: Chromatic blur + defocus
    rgb = chromatic_psf_blur(rgb, na=na, wavelengths_nm=wls,
                              defocus_nm=defocus_nm, pixel_size_nm=pixel_size_nm)

    # Step 3: Illumination gradient
    illum = illumination_gradient(thickness_map_nm.shape, pixel_size_nm=pixel_size_nm, mode=mode, seed=seed)
    rgb = apply_illumination(rgb, illum)

    # Step 4: Scale to photons and add sensor noise
    rgb_photons = rgb * photon_flux
    rgb_dn = apply_sensor_noise(rgb_photons, seed=seed)
    
    # rgb_dn is RGB order. Convert to BGR for standard cv2 saving.
    return cv2.cvtColor(rgb_dn, cv2.COLOR_RGB2BGR)

def optical_image_reference(canvas: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Wraps simulation for the clean reference image (1nm px)"""
    rng = np.random.default_rng(seed)
    
    # Base oxide thickness variation across the die
    thickness_map = rng.normal(loc=20.0, scale=0.5, size=canvas.shape)
    
    return simulate_rgb_wafer_image(
        height_map_nm=canvas,
        thickness_map_nm=thickness_map,
        pixel_size_nm=1.0,
        defocus_nm=0.0, # Perfect focus for reference
        na=0.9,
        photon_flux=50000, # High dose reference
        mode='brightfield',
        seed=seed
    )

def optical_image_search(canvas: np.ndarray, seed: int | None = None, defocus_nm: float = 30.0, flux: float = 10000) -> np.ndarray:
    """Wraps simulation for the noisy/defocused search image (10nm px)"""
    rng = np.random.default_rng(seed)
    
    # PERFORMANCE OPTIMIZATION:
    # Instead of doing 10000x10000 FFTs and then resizing at the end, 
    # we physically downsample the topography first, reducing calculations by 100x.
    h, w = canvas.shape
    canvas_10nm = cv2.resize(canvas, (w // 10, h // 10), interpolation=cv2.INTER_AREA)

    # Thicker variation for wider FOV
    thickness_map = rng.normal(loc=20.0, scale=1.5, size=canvas_10nm.shape)
    
    rgb_dn = simulate_rgb_wafer_image(
        height_map_nm=canvas_10nm,
        thickness_map_nm=thickness_map,
        pixel_size_nm=10.0,  # Simulate natively at 10nm resolution
        defocus_nm=defocus_nm, # Out of focus during fast scanning
        na=0.9,
        photon_flux=flux, # Lower dose for search scan
        mode='brightfield',
        seed=seed
    )
    
    return rgb_dn
