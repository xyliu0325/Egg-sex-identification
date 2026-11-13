"""Mean RGB intensities at the centers of a 3 x 6 spot array. Install: python -m pip install opencv-python numpy openpyxl
Running without --input uses DIR_PATH below. Default coordinates correspond to a 601 x 300 reference image.
Images must have consistent cropping and orientation; calibrate each group separately if spot positions differ.
Initial manual calibration: python extract_rgb.py --input "path/to/images" --calibrate
Reuse spatial calibration: python extract_rgb.py --input "path/to/images" --calibration "path/to/images/RGB_results_MAD35/calibration.json"
Apply one pass of median/MAD outlier filtering to the original 8-bit pixel values, without color correction.
The robust scale estimate for each channel is max(1.4826*MAD, 1.0); the rejection threshold is 3.5 times this value.
Reject the entire pixel if any channel exceeds its threshold; filtering is not iterative.
Inspection previews show the original image on the left and red sampling outlines on the right; magenta marks rejected pixels.
Excel output contains only Filename, Spot, Red, Green, and Blue; RGB values are filtered means rounded and displayed to two decimal places.
"""
import argparse
import json
import re
from pathlib import Path
import cv2
import numpy as np
from openpyxl import Workbook

DIR_PATH = Path(r"F:\20261113\Batch4\DAY5")
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
DEFAULT_CORNERS = np.array([[54/601, 57/300], [525/601, 57/300],
                            [54/601, 244/300], [525/601, 244/300]])
RADIUS_FRACTION = 0.13
ROBUST_K = 3.5
MIN_ROBUST_SIGMA = 1.0
REVIEW_FRACTION = 0.20


def reject_outliers(pixels):
    """Accept an N x 3 RGB array and return a pixel retention mask and filtering statistics."""
    median = np.median(pixels, axis=0)
    mad = np.median(np.abs(pixels-median), axis=0)
    sigma = np.maximum(1.4826*mad, MIN_ROBUST_SIGMA)
    limits = ROBUST_K*sigma
    keep = np.all(np.abs(pixels-median) <= limits, axis=1)
    return keep, median, mad, limits


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', path.name)]


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Image decoding failed; verify the input file.')
    if image.dtype != np.uint8:
        raise ValueError('An 8-bit image is required; convert 16-bit images using a consistent conversion procedure before processing.')
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError('An RGB color image is required.')
    if image.shape[2] == 4:
        if np.any(image[:, :, 3] != 255):
            raise ValueError('Transparent pixels detected; select a background and convert the image to opaque RGB before processing.')
        image = image[:, :, :3].copy()
    return image


def grid_centers(corners):
    tl, tr, bl, br = corners
    return np.array([(1-v)*((1-u)*tl+u*tr)+v*((1-u)*bl+u*br)
                     for v in (0, .5, 1) for u in np.linspace(0, 1, 6)]).reshape(3, 6, 2)


def calibrate(image):
    h, w = image.shape[:2]
    scale = min(1., 1200/w, 750/h)
    view = cv2.resize(image, (round(w*scale), round(h*scale)))
    points = []
    title = 'Centers: TL, TR, BL, BR; right-click=undo; Enter=OK; Esc=cancel'
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse)
    print('Select the upper-left, upper-right, lower-left, and lower-right spot centers, in that order. Select spot centers rather than image corners. Right-click to undo; press Enter to confirm.')
    try:
        while True:
            shown = view.copy()
            for i, (x, y) in enumerate(points, 1):
                cv2.circle(shown, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(shown, str(i), (x+6, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            cv2.imshow(title, shown)
            key = cv2.waitKey(30) & 255
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError('Calibration cancelled.')
            if key in (10, 13) and len(points) == 4:
                return np.array(points, dtype=float) / [view.shape[1], view.shape[0]]
    finally:
        cv2.destroyAllWindows()


def measure(image, corners, fraction):
    h, w = image.shape[:2]
    grid = grid_centers(corners * [w, h])
    distances = np.concatenate([np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                                np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()])
    radius = max(1, round(float(distances.min()) * fraction))
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = xx*xx + yy*yy <= radius*radius
    preview, records = image.copy(), []
    for row in range(3):
        for col in range(6):
            x, y = np.rint(grid[row, col]).astype(int)
            number = row*6+col+1
            if x-radius < 0 or y-radius < 0 or x+radius >= w or y+radius >= h:
                raise ValueError(f'The sampling region for spot {number} extends beyond the image boundaries; recalibrate the spot positions.')
            roi = image[y-radius:y+radius+1, x-radius:x+radius+1]
            pixels = roi[mask][:, ::-1].astype(float)
            keep, median, mad, limits = reject_outliers(pixels)
            total = len(pixels)
            kept = int(keep.sum())
            deleted = total-kept
            ratio = deleted/total
            filtered_mean = pixels[keep].mean(axis=0) if kept else np.full(3, np.nan)
            filtered_std = pixels[keep].std(axis=0) if kept else np.full(3, np.nan)
            status = 'No pixels retained; result invalid' if not kept else ('More than 20% of pixels rejected; review required' if ratio > REVIEW_FRACTION else 'OK')
            records.append([number, row+1, col+1, x, y, radius, total, kept, deleted,
                            ratio*100, *pixels.mean(axis=0), *filtered_mean,
                            *pixels.std(axis=0), *filtered_std, *median, *mad, *limits, status])
            rejected_mask = np.zeros_like(mask)
            rejected_mask[mask] = ~keep
            preview_roi = preview[y-radius:y+radius+1, x-radius:x+radius+1]
            preview_roi[rejected_mask] = (255, 0, 255)
            cv2.circle(preview, (x, y), radius+1, (0, 0, 255), 1)
            cv2.putText(preview, str(number), (x-radius, y-radius-4),
                        cv2.FONT_HERSHEY_SIMPLEX, max(.35, w/1500), (0, 0, 255), 1, cv2.LINE_AA)
    return records, preview


def comparison_preview(image, annotated):
    """Compare original colors and rejected pixel locations side by side; use nearest-neighbor enlargement to preserve pixel positions."""
    scale = 3 if image.shape[1] <= 800 else 1
    paired = np.concatenate([image, annotated], axis=1)
    if scale != 1:
        paired = cv2.resize(paired, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    width = paired.shape[1]
    canvas = np.full((paired.shape[0]+64, max(width, 900), 3), 255, dtype=np.uint8)
    canvas[64:, :width] = paired
    cv2.putText(canvas, 'LEFT: original | RIGHT: red = ROI, magenta = rejected pixels',
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'One pass: abs(RGB - median) > 3.5 * max(1.4826*MAD, 1.0); any channel rejects pixel',
                (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_excel(results, path):
    """Preserve measurement precision until export, then round to two decimal places and apply the Excel display format."""
    book = Workbook()
    sheet = book.active
    sheet.title = 'RGB'
    sheet.append(['Filename', 'Spot', 'Red', 'Green', 'Blue'])
    for record in results:
        rgb = [round(float(v), 2) if np.isfinite(v) else None for v in record[14:17]]
        sheet.append([record[0], int(record[1]), *rgb])
    for row in sheet.iter_rows(min_row=2, min_col=3, max_col=5):
        for cell in row:
            cell.number_format = '0.00'
    sheet.column_dimensions['A'].width = 32
    for column in ('B', 'C', 'D', 'E'):
        sheet.column_dimensions[column].width = 12
    sheet.freeze_panes = 'A2'
    book.save(path)


def main():
    parser = argparse.ArgumentParser(description='Extract mean RGB intensities from the centers of 18 circular spots.')
    parser.add_argument('--input', type=Path, default=DIR_PATH)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--radius-fraction', type=float, default=RADIUS_FRACTION)
    args = parser.parse_args()
    if not 0 < args.radius_fraction <= .20:
        parser.error('radius-fraction must be in the interval (0, 0.20].')
    if not args.input.is_dir():
        parser.error(f'Input directory does not exist: {args.input}')
    files = sorted([p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS], key=natural_key)
    if not files:
        parser.error('No supported image files were found in the input directory.')
    corners = DEFAULT_CORNERS.copy()
    if args.calibration:
        config = json.loads(args.calibration.read_text(encoding='utf-8'))
        corners = np.asarray(config['normalized_corners'], dtype=float)
    if args.calibrate:
        corners = calibrate(read_image(files[0]))
    if corners.shape != (4, 2) or not np.isfinite(corners).all() or np.any((corners < 0) | (corners > 1)):
        parser.error('Calibration must contain four points with normalized coordinates in the interval [0, 1].')
    grid = grid_centers(corners)
    if min(np.linalg.norm(np.diff(grid, axis=1), axis=2).min(), np.linalg.norm(np.diff(grid, axis=0), axis=2).min()) < .01:
        parser.error('Calibration points are too close together; select the spot centers again.')
    out = args.input / 'RGB_results_MAD35'
    previews = out / 'previews'
    previews.mkdir(parents=True, exist_ok=True)
    (out / 'calibration.json').write_text(json.dumps({
        'normalized_corners': corners.tolist(), 'radius_fraction': args.radius_fraction,
        'robust_k': ROBUST_K, 'min_robust_sigma': MIN_ROBUST_SIGMA,
        'note': 'Spot centers in upper-left, upper-right, lower-left, and lower-right order. Applicable only to images with consistent cropping.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    results, errors = [], []
    for path in files:
        try:
            image = read_image(path)
            rows, annotated = measure(image, corners, args.radius_fraction)
            preview = comparison_preview(image, annotated)
            ok, data = cv2.imencode('.png', preview)
            if not ok:
                raise OSError('Preview image encoding failed.')
            data.tofile(str(previews / (path.name + '_ROI.png')))
            results.extend([[path.name, *row] for row in rows])
            print(f'Processed: {path.name}; 18 spots')
            for row in rows:
                if row[-1] != 'OK':
                    print(f'  Spot {row[0]}: {row[-1]}; {row[9]:.1f}% of pixels rejected')
        except (ValueError, OSError, cv2.error) as exc:
            errors.append(f'{path.name}: {exc}')
            print(f'Skipped: {errors[-1]}')
    save_excel(results, out / 'RGB_results.xlsx')
    (out / 'errors.txt').write_text('\n'.join(errors) if errors else 'All images were processed successfully.', encoding='utf-8')
    print(f'Completed: {len(results)//18}/{len(files)} images; output directory: {out}')
    print('Previews: original image on the left; red outlines indicate sampling regions and magenta marks rejected pixels on the right. Inspect the spot centers and the colors of rejected pixels.')
    print('Repeated runs overwrite output files with the same names. std denotes variation among pixels within a sampling region, not variability across experimental replicates.')


if __name__ == '__main__':
    main()
