import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
import re

# ------------------------------------------ CONFIG ------------------------------------------
INPUT_FOLDER = Path(r"C:\Users\Ethan\Desktop\New_Poop_Analysis\Original Images\DSS 3 (pilot rat study)\Rat 16")
CSV_OUTPUT = Path(r"C:\Users\Ethan\Desktop\New_Poop_Analysis\CSV_Outputs\DSS 3_well_intensities.csv")
ROI_OUTPUT_FOLDER = Path(r"C:\Users\Ethan\Desktop\New_Poop_Analysis\Individual Well Images\DSS 3_well_ROIs")
ROI_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Save saturated full images
SAT_OUTPUT_FOLDER = Path(r"C:\Users\Ethan\Desktop\poop_analysis\saturated_full_images")
SAT_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Saturation factor to apply for saving (does NOT affect analysis)
SAT_FACTOR = 4.0  # Change if needed (1.0 = original, 0 = grayscale, >1 = more saturated))

# Resolution for circle detection display
DISP_W, DISP_H = 1200, 800

# ---- CIRCLE DETECTION PARAMS ----
BLUR = 3
DP = 1.0
MINDIST = 240
PARAM1 = 57
PARAM2 = 30
MINR = 99
MAXR = 118

# Valid image types
VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
# --------------------------------------------------------------------------------------------


def scale_for_detection(img, max_w=DISP_W, max_h=DISP_H):
    """
    Downscale image for circle detection only.
    This improves performance and stabilizes HoughCircles.
    Returns the resized image and the scale factor.
    """
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return disp, scale


def detect_wells_hough(disp_bgr):
    """
    Detect circular wells using HoughCircles on a scaled image.
    Detection parameters are fixed and tuned for consistency.
    """
    gray = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2GRAY)

    # Ensure Gaussian kernel size is odd
    k = BLUR if BLUR % 2 == 1 else BLUR + 1
    gray = cv2.GaussianBlur(gray, (k, k), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MINDIST,
        param1=PARAM1,
        param2=PARAM2,
        minRadius=MINR,
        maxRadius=MAXR
    )

    if circles is None:
        return None

    # Return circles as integer (x, y, r)
    circles = np.round(circles[0]).astype(int)
    return circles


def pick_and_order_12(circles, ncols=4):
    """
    Select exactly 12 wells and enforce consistent ordering.
    Wells are sorted top-to-bottom (rows), then left-to-right (columns).
    """
    if circles is None or len(circles) == 0:
        return None

    # If extra circles are detected, keep the 12 largest
    if len(circles) > 12:
        circles = circles[np.argsort(circles[:, 2])[::-1]][:12]

    # Sort by vertical position to form rows
    circles = circles[np.argsort(circles[:, 1])]

    # If exactly 12, sort into 3 rows × 4 columns
    if len(circles) == 12:
        rows = []
        for i in range(0, 12, ncols):
            row = circles[i:i+ncols]
            row = row[np.argsort(row[:, 0])]  # sort each row left-to-right
            rows.append(row)
        circles = np.vstack(rows)

    return circles


def circles_to_full_res(circles_disp, scale):
    """
    Convert detected circle coordinates from scaled image
    back to original full-resolution coordinates.
    """
    if circles_disp is None:
        return None
    circles_full = circles_disp.astype(np.float32)
    circles_full[:, :3] /= scale
    return np.round(circles_full).astype(int)


def compute_green_intensity(pil_img, mask_bool):
    """
    Compute green fluorescence intensity inside a circular mask.
    Metric used: SUM(S * V) for pixels within HSV green range.
    """
    hsv = np.asarray(pil_img.convert("HSV")).astype(np.float32)

    H = hsv[:, :, 0] * 360.0 / 255.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0

    # green fluorescence mask
    green_mask = (H >= 90) & (H <= 150) & (S >= 0.1) & (V >= 0.2)
    final_mask = green_mask & mask_bool

    if not np.any(final_mask):
        return 0.0

    return float(np.sum(S[final_mask] * V[final_mask]))


def crop_circle_roi_and_mask(pil_img, cx, cy, r):
    """
    Crop a square ROI around a detected well and apply a circular mask.
    Pixels outside the well are set to black.
    """
    arr = np.array(pil_img)
    H, W = arr.shape[:2]

    # Bounding box for square crop
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, W - 1)
    y2 = min(cy + r, H - 1)

    crop = arr[y1:y2+1, x1:x2+1].copy()

    # Create circular mask in cropped coordinates
    hh, ww = crop.shape[:2]
    yy, xx = np.ogrid[:hh, :ww]
    ccx = cx - x1
    ccy = cy - y1
    mask = (xx - ccx) ** 2 + (yy - ccy) ** 2 <= r ** 2

    crop[~mask] = 0
    return Image.fromarray(crop), mask


def full_image_circle_mask(shape_hw, cx, cy, r):
    """
    Generate a boolean circular mask over the full image.
    Used for intensity calculation.
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask.astype(bool)


def save_saturated_full_image(pil_img, save_path, sat_factor):
    """
    Save a saturated version of the full image for visualization.
    This does NOT affect detection or analysis.
    """
    enhancer = ImageEnhance.Color(pil_img)
    sat_img = enhancer.enhance(float(sat_factor))
    sat_img.save(save_path)


def parse_day_image(filename):
    """
    Extract day and image number from filenames like:
    DSS3 R16 T-3.2.png
    DSS3 R16 T3.1.png
    """
    match = re.search(r"T(-?\d+)\.(\d+)", filename)
    if match:
        day = int(match.group(1))
        image_num = int(match.group(2))
        return day, image_num
    else:
        return None, None

# ---------------- PROCESS ALL IMAGES ----------------
results = []

# Gather all valid images in input directory
image_files = sorted([p for p in INPUT_FOLDER.iterdir() if p.suffix.lower() in VALID_EXT])

for img_path in image_files:
    print(f"Processing: {img_path.name}")
    
    day, image_num = parse_day_image(img_path.name)
    if day is None:
        print("  !! Filename did not match pattern (skipping):", img_path.name)
        continue

    # Load image for OpenCV-based detection
    cv_full = cv2.imread(str(img_path))
    if cv_full is None:
        print(f"  !! OpenCV couldn't read {img_path.name}, skipping")
        continue

    # Load original image for ROI extraction and intensity analysis
    pil_full = Image.open(img_path).convert("RGB")

    # Save saturated full image (visual reference only)
    sat_name = f"{img_path.stem}_sat{SAT_FACTOR:g}{img_path.suffix}"
    sat_path = SAT_OUTPUT_FOLDER / sat_name
    save_saturated_full_image(pil_full, sat_path, SAT_FACTOR)

    # Detect wells on scaled image, then map to full resolution
    disp, scale = scale_for_detection(cv_full)
    circles_disp = detect_wells_hough(disp)
    circles_disp = pick_and_order_12(circles_disp, ncols=4)
    circles_full = circles_to_full_res(circles_disp, scale)

    # Create output folder for this image's ROIs
    out_folder = ROI_OUTPUT_FOLDER / img_path.stem
    out_folder.mkdir(parents=True, exist_ok=True)

    # Initialize per-well intensity list
    intensities = [np.nan] * 12

    if circles_full is None:
        print("  !! No circles detected")
    else:
        n = min(len(circles_full), 12)

        for i in range(n):
            cx, cy, r = circles_full[i]

            # Compute green intensity using full-image mask
            mask_bool = full_image_circle_mask(cv_full.shape[:2], cx, cy, r)
            intensities[i] = compute_green_intensity(pil_full, mask_bool)

            # Save cropped, masked well ROI
            roi_img, _ = crop_circle_roi_and_mask(pil_full, cx, cy, r)
            roi_path = out_folder / f"well_{i+1:02d}.png"
            roi_img.save(roi_path)

    # Store day + image number alongside well intensities
    results.append([img_path.name, day, image_num] + intensities)

# ---------------- SAVE CSV ----------------
colnames = ["image", "Day", "ImageNum"] + [f"well_{i:02d}" for i in range(1, 13)]
df = pd.DataFrame(results, columns=colnames)
df.to_csv(CSV_OUTPUT, index=False)

print("\n---------------------------------------------")
print(f"Done! CSV saved to: {CSV_OUTPUT}")
print(f"Well ROIs saved under: {ROI_OUTPUT_FOLDER}")
print(f"Saturated full images saved under: {SAT_OUTPUT_FOLDER} (factor={SAT_FACTOR})")
print("---------------------------------------------")