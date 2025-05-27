import os
import numpy as np
import cv2
from scipy.signal import find_peaks
from skimage import io
import glob # For easily finding all .png files in a directory

from annotator.segmentation.utils import load_images_from_folder


def gen_bounding_boxes(det,peaks, lineheight_baseline_percentile, binarize_threshold):
  img = np.uint8(det * 255)
  _, img1 = cv2.threshold(img, binarize_threshold, 255, cv2.THRESH_BINARY)

  # Find contours
  contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  bounding_boxes = []
  max_height = np.percentile(peaks[1:]-peaks[:-1],lineheight_baseline_percentile)
  # Extract bounding boxes from contours
  for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if h<=max_height:
          bounding_boxes.append((x, y, w, h))
      else:
          n_b = np.int32(np.ceil(h/max_height))
          # Calculate the height of each box
          equal_height = h // n_b

          # Calculate the height adjustment needed for the last box to ensure total height is covered
          height_adjustment = h - (equal_height * n_b)

          for i in range(n_b):
              new_y = y + (i * equal_height)
              # Adjust the height of the last box if necessary
              box_height = equal_height + (height_adjustment if i == n_b - 1 else 0)
              bounding_boxes.append((x, new_y, w, box_height))

  return bounding_boxes

def assign_lines(bounding_boxes,det):

  ys = det.sum(axis=1)
  thres = 0.5 * ys.max()
  peaks, _ = find_peaks(ys, height=thres,distance=det.shape[0]/100,width=5)

  lines = []
  xs = det.sum(axis = 0)
  thres = 0.5 * xs.max()
  xpeaks, _ = find_peaks(xs, height=thres)
  ys1 = det[:,xpeaks[0]:xpeaks[0]+100].sum(axis=1)
  thres = 0.5 * ys1.max()
  p1, _ = find_peaks(ys1, height=thres,distance=det.shape[0]/100,width=5)
  ys2 = det[:,xpeaks[-1]-100:xpeaks[-1]].sum(axis=1)
  thres = 0.5 * ys2.max()
  p2, _ = find_peaks(ys2, height=thres,distance=det.shape[0]/100,width=5)
  xmid = int((xpeaks[0]+xpeaks[-1])/2)
  ys3 = det[:,xmid-50:xmid+50].sum(axis=1)
  thres = 0.5 * ys3.max()
  p3, _ = find_peaks(ys3, height=thres,distance=det.shape[0]/100,width=5)
  if(peaks[0]-p1[0]>det.shape[0]/12):
    p1 = np.copy(p1[1:])
  p= min(p1, p2, p3, key=len)
  l = len(p)
  if(len(p1)>=l+1):
    k = len(p1) - len(p)
    ind = np.argmin(np.abs(p1[:k+1] - p[0]))
    peaks1 = p1[ind:l+ind]
  else:
    peaks1 = p1

  if(len(p2)>=l+1):
    k = len(p2) - len(p)
    ind = np.argmin(np.abs(p2[:k+1] - p[0]))
    peaks2 = p2[ind:l+ind]
  else:
    peaks2 = p2

  if(len(p3)>=l+1):
    k = len(p3) - len(p)
    ind = np.argmin(np.abs(p3[:k+1] - p[0]))
    peaks3 = p3[ind:l+ind]
  else:
    peaks3 = p3

  for box in bounding_boxes:
      x, y, _, h = box
      mid_y = y + h / 2  # Midpoint of the y-dimension
      wt1 = np.abs(x - xpeaks[0])
      wt2 = np.abs(x - xpeaks[-1])
      wt3 = np.abs(x - xmid)
      if x<=xmid:
        peaks = wt3*peaks1/(wt1+wt3)+wt1*peaks3/(wt1+wt3)
      else:
        peaks = wt3*peaks2/(wt2+wt3)+wt2*peaks3/(wt2+wt3)

      # Calculate the absolute difference between mid_y and each peak, then find the index of the minimum difference
      c_index = np.argmin(np.abs(peaks - mid_y))
      if(np.abs(mid_y-peaks[c_index])>20):
          c_index=-1
      lines.append(c_index)
  return lines,peaks1

def crop_img(img):
    sum_rows = np.sum(img, axis=1)
    sum_cols = np.sum(img, axis=0)

    # Find indices where sum starts to vary for rows
    row_start = np.where(sum_rows != sum_rows[0])[0][0] if np.any(sum_rows != sum_rows[0]) else 0
    row_end = np.where(sum_rows != sum_rows[-1])[0][-1] if np.any(sum_rows != sum_rows[-1]) else len(sum_rows) - 1

    # Find indices where sum starts to vary for columns
    col_start = np.where(sum_cols != sum_cols[0])[0][0] if np.any(sum_cols != sum_cols[0]) else 0
    col_end = np.where(sum_cols != sum_cols[-1])[0][-1] if np.any(sum_cols != sum_cols[-1]) else len(sum_cols) - 1

    # Crop the image using the identified indices
    return np.copy(img[row_start:row_end+1, col_start:col_end+1])

def gen_line_images(img2,peaks,bounding_boxes,lines, lineheight_baseline_percentile):
  # change here
#   global lineheight_baseline_percentile
  line_images=[]
  max_height_line = np.percentile(peaks[1:]-peaks[:-1],lineheight_baseline_percentile)
  pad=int(max_height_line*0.2)
  for l in range(len(peaks)):
      # Filter bounding boxes for the current label
      filtered_boxes = [box for box, idx in zip(bounding_boxes, lines) if idx == l]

      if not filtered_boxes:
          continue

      # Calculate the total width and maximum height for the new image
      total_width = max(x for x, _,_, _ in filtered_boxes) + 500  # 10 pixels padding on each side
      max_height = max(h for _, _, _, h in filtered_boxes) + 250  # 5 pixels padding top and bottom
      miny = min(y for _, y,_, _ in filtered_boxes)
      # Create an empty image for this label
      new_img = np.ones((max_height, total_width), dtype=np.uint8)*np.int32(np.median(img2))

      for box in filtered_boxes:
          x, y, w, h = box
          blob = img2[y-pad:y+h+pad, x-10:x+w+10]
          new_img[y-miny:y-miny+h+2*pad,x-10:x+w+10]=blob
      line_images.append(crop_img(new_img))

  return line_images



def load_saved_heatmaps_jpg(heatmap_folder_path):
    """
    Loads heatmaps that were saved as .jpg files by multiplying by 255
    and using cv2.imwrite.

    Args:
        heatmap_folder_path (str): The path to the folder where .jpg heatmaps are stored.
                                   e.g., "instance/manuscripts/your_m_name/heatmaps"

    Returns:
        tuple: (loaded_heatmaps, heatmap_filenames)
            - loaded_heatmaps (list): A list of NumPy arrays, each representing a heatmap
                                      with float values in the range [0, 1].
            - heatmap_filenames (list): A list of corresponding filenames (e.g., "image1.jpg").
    """
    if not os.path.isdir(heatmap_folder_path):
        print(f"Error: Heatmap folder not found at {heatmap_folder_path}")
        return [], []

    loaded_heatmaps = []
    heatmap_filenames = []

    # Find all .jpg files in the specified folder
    # Sort them to ensure a consistent order if that matters for your application
    jpg_files = sorted(glob.glob(os.path.join(heatmap_folder_path, "*.jpg"))) # Changed to *.jpg

    if not jpg_files:
        print(f"No .jpg files found in {heatmap_folder_path}") # Updated message
        return [], []

    print(f"Found {len(jpg_files)} heatmap .jpg files to load.")

    for file_path in jpg_files:
        # Load the image in grayscale mode
        # cv2.imread by default loads in BGR. For single channel (grayscale) like heatmaps:
        img_uint8 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img_uint8 is None:
            print(f"Warning: Could not load image {file_path}. Skipping.")
            continue

        # Convert to float and scale back to [0, 1] range
        # The original data was likely float before being multiplied by 255 and saved as uint8
        heatmap_float = img_uint8.astype(np.float32) / 255.0

        loaded_heatmaps.append(heatmap_float)
        heatmap_filenames.append(os.path.basename(file_path))

    return loaded_heatmaps, heatmap_filenames


def segment_lines(folder_path, lineheight_baseline_percentile=80, binarize_threshold=100):
    print(folder_path)
    #m_name = folder_path.split('/')[-2]
    m_name = os.path.basename(os.path.dirname(folder_path))



    # LOAD HEATMAP
    inp_images, file_names = load_images_from_folder(folder_path)
    out_images,_ = load_saved_heatmaps_jpg(f'instance/manuscripts/{m_name}/heatmaps')


    # ALGORITHM
    for det,image,file_name in zip(out_images,inp_images,file_names):
        print(file_name)
        ys = det.sum(axis=1)
        thres = 0.5 * ys.max()
        try:
            peaks, _ = find_peaks(ys, height=thres,distance=det.shape[0]/100,width=5)
            bounding_boxes = gen_bounding_boxes(det,peaks, lineheight_baseline_percentile, binarize_threshold)
            img2 = cv2.cvtColor(cv2.resize(image, det.shape[::-1]), cv2.COLOR_BGR2GRAY)
            
            lines,peaks1 = assign_lines(bounding_boxes,det)
            line_images = gen_line_images(img2,peaks1,bounding_boxes,lines, lineheight_baseline_percentile)

            if os.path.exists(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}') == False:
                os.makedirs(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}')
            for i in range(len(line_images)):
                cv2.imwrite(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}/line{i+1:03d}.jpg',line_images[i])
        except:
            print("segmentation fails")
            with open(f'instance/manuscripts/{m_name}/points-2D/failures.txt', 'a') as file:
                file.write(f"{file_name}")

            if os.path.exists(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}') == False:
                os.makedirs(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}')
            black_image = np.zeros((50, 900, 3), dtype=np.uint8)
            for i in range(5):
                cv2.imwrite(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}/line{i+1:03d}.jpg',black_image)
    
