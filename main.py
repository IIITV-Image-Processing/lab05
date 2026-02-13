import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import skew, kurtosis
import os
import pandas as pd


GAUSSIAN_BLUR_KSIZE = 5
MEDIAN_FILTER_KSIZE = 5
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
AVERAGING_KSIZE = 5
LOWPASS_CUTOFF_RATIO = 0.3
NOTCH_RADIUS = 10
NOTCH_THRESHOLD = 0.85

# config

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output"

os.makedirs(f"{OUTPUT_FOLDER}/spatial", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/frequency", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/histograms", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/fft", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/comparison", exist_ok=True)

# noise detection implementation

def detect_noise(image):
    img = image.astype(np.float64)

    # Salt & Pepper detection
    salt = np.sum(img > 250)
    pepper = np.sum(img < 5)
    sp_ratio = (salt + pepper) / img.size
    if sp_ratio > 0.02:
        return "salt_and_pepper"

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.log(1 + np.abs(fshift))
    mag_norm = mag / mag.max()

    cy, cx = mag_norm.shape[0] // 2, mag_norm.shape[1] // 2
    mask = np.ones_like(mag_norm, dtype=bool)
    mask[cy-10:cy+10, cx-10:cx+10] = False

    peaks = np.sum(mag_norm[mask] > NOTCH_THRESHOLD)
    if peaks > 20:
        return "periodic"

    mean_val = np.mean(img)
    var_val = np.var(img)
    sk = skew(img.flatten())

    coeff_var = np.sqrt(var_val) / (mean_val + 1e-10)
    if coeff_var > 0.5 and abs(sk) > 1:
        return "speckle"

    return "gaussian"

# spatial filter implementation

def apply_spatial_filter(image, noise_type):
    if noise_type == "gaussian":
        return cv2.GaussianBlur(image, (GAUSSIAN_BLUR_KSIZE, GAUSSIAN_BLUR_KSIZE), 0)

    elif noise_type == "salt_and_pepper":
        return cv2.medianBlur(image, MEDIAN_FILTER_KSIZE)

    elif noise_type == "speckle":
        return cv2.bilateralFilter(
            image, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
        )

    elif noise_type == "periodic":
        return cv2.GaussianBlur(image, (3, 3), 0)

    return image.copy()

# frequency filter implementation

def apply_frequency_filter(image, noise_type):
    img = image.astype(np.float64)
    rows, cols = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    if noise_type == "periodic":
        mag = np.log(1 + np.abs(fshift))
        mag_norm = mag / mag.max()

        H = np.ones((rows, cols))
        cy, cx = rows // 2, cols // 2

        ys, xs = np.where(mag_norm > NOTCH_THRESHOLD)

        for y, x in zip(ys, xs):
            if abs(y - cy) > 10 or abs(x - cx) > 10:
                rr, cc = np.ogrid[:rows, :cols]
                mask = (rr - y) ** 2 + (cc - x) ** 2 <= NOTCH_RADIUS ** 2
                H[mask] = 0

        filtered = fshift * H

    else:
        # Gaussian Low-pass
        cy, cx = rows // 2, cols // 2
        rr, cc = np.ogrid[:rows, :cols]
        dist = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)

        cutoff = LOWPASS_CUTOFF_RATIO * min(rows, cols) / 2
        H = np.exp(-(dist ** 2) / (2 * cutoff ** 2))

        filtered = fshift * H

    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    return img_back

# metrics calculation

def calculate_metrics(reference, denoised):
    ref = reference.astype(np.float64)
    den = denoised.astype(np.float64)

    mse = np.mean((ref - den) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    ssim_val = ssim(reference, denoised, data_range=255)

    return mse, psnr, ssim_val

# Visualization
def save_histogram(image, name):
    plt.figure(figsize=(6,4))
    plt.hist(image.flatten(), bins=256)
    plt.title("Intensity Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.savefig(f"{OUTPUT_FOLDER}/histograms/{name}_hist.png")
    plt.close()

def save_fft_spectrum(image, name):
    f = np.fft.fft2(image.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.log(1 + np.abs(fshift))

    plt.figure(figsize=(6,6))
    plt.imshow(mag, cmap='gray')
    plt.title("FFT Magnitude Spectrum")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_FOLDER}/fft/{name}_fft.png")
    plt.close()

def save_comparison_image(original, spatial, frequency, name, noise_type):
    # FFT magnitude
    f = np.fft.fft2(original.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.log(1 + np.abs(fshift))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{name} - Noise: {noise_type}", fontsize=14)

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original (Noisy)")
    axes[0].axis("off")

    axes[1].imshow(spatial, cmap='gray')
    axes[1].set_title("Spatial Filtered")
    axes[1].axis("off")

    axes[2].imshow(frequency, cmap='gray')
    axes[2].set_title("Frequency Filtered")
    axes[2].axis("off")

    axes[3].imshow(mag, cmap='gray')
    axes[3].set_title("FFT Spectrum")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(f"{name}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():

    results = []

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".png")]

    for name in image_files:
        path = os.path.join(INPUT_FOLDER, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        print(f"\nProcessing {name}")

        noise_type = detect_noise(img)
        print(f"Detected Noise: {noise_type}")

        spatial = apply_spatial_filter(img, noise_type)
        frequency = apply_frequency_filter(img, noise_type)

        cv2.imwrite(f"{OUTPUT_FOLDER}/spatial/{name}", spatial)
        cv2.imwrite(f"{OUTPUT_FOLDER}/frequency/{name}", frequency)

        save_histogram(img, name.replace(".png", ""))
        save_fft_spectrum(img, name.replace(".png", ""))

        mse_s, psnr_s, ssim_s = calculate_metrics(img, spatial)
        mse_f, psnr_f, ssim_f = calculate_metrics(img, frequency)

        better = "Spatial" if psnr_s > psnr_f else "Frequency"

        print(f"Better Method: {better}")

        results.append([
            name, noise_type,
            psnr_s, psnr_f,
            ssim_s, ssim_f,
            better
        ])

        save_comparison_image(
            img,
            spatial,
            frequency,
            name.replace(".png", ""),
            noise_type
        )

    df = pd.DataFrame(results, columns=[
        "Image", "Noise Type",
        "PSNR Spatial", "PSNR Frequency",
        "SSIM Spatial", "SSIM Frequency",
        "Better Method"
    ])

    print("\nFinal Results:\n")
    print(df)

    df.to_csv(f"{OUTPUT_FOLDER}/results.csv", index=False)

    print("\nAll outputs saved inside 'output' folder.")
    print("\nComparison images saved in 'output/comparison' folder.")
    


if __name__ == "__main__":
    main()
