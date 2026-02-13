# Image Noise Detection and Filtering

## Description

This project implements an automated image noise detection and filtering system that identifies different types of noise (Gaussian, Salt & Pepper, Speckle, Periodic) and applies appropriate spatial and frequency domain filters. The system processes images, generates comparison visualizations, computes quality metrics (PSNR, SSIM), and produces a comprehensive analysis report.

## How to Run

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # venv\Scripts\activate   # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your noisy images:**
   - Place PNG images in the `images/` folder

4. **Run the program:**
   ```bash
   python main.py
   ```

4. **View results:**
   - Filtered images: `output/spatial/` and `output/frequency/`
   - Comparison images: `output/comparison/`
   - Histograms: `output/histograms/`
   - FFT spectrums: `output/fft/`
   - Metrics report: `output/results.csv`
