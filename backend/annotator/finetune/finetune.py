# import sys
# import os
# import csv
# import shutil
# import random
# import yaml
# import pandas as pd
# import logging # Import the logging module

# from datetime import datetime
# from flask import current_app

# from annotator.finetune.utils import AttrDict
# from annotator.finetune.train import train
# from model.models import db, UserAnnotationLog
# from database.connection import get_db
# from model.manuscriptmodel import AnnotationLog
# from sqlalchemy.exc import SQLAlchemyError


# # --- Configure logging at the top of the file ---
# # This ensures a consistent logging setup throughout the script.
# # You can adjust the level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# # and format as needed. For debugging, INFO or DEBUG is usually best.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Ensure stdout/stderr are configured for UTF-8 (good practice, keep it)
# sys.stdout.reconfigure(encoding='utf-8')
# sys.stderr.reconfigure(encoding='utf-8')


# def get_config(file_path, manuscript_name, selected_model, model_name):
#     logger.info(f"Loading configuration from: {file_path}")
#     try:
#         with open(file_path, "r", encoding="utf-8") as stream:
#             opt = yaml.safe_load(stream)
#         opt = AttrDict(opt)
#         opt.character = opt.number + opt.symbol + opt.lang_char
#         opt.manuscript_name = manuscript_name
#         opt.saved_model = os.path.join(
#             os.path.join(current_app.config['DATA_PATH']), "models", "recognition", selected_model,
#         )
#         opt.model_name = model_name
        
#         # --- Logging config details ---
#         logger.info(f"Config loaded for manuscript: '{manuscript_name}' with model: '{selected_model}' (output name: '{model_name}')")
#         logger.debug(f"Full config opt object: {opt}")

#         save_model_dir = f"./saved_models/{opt.model_name}"
#         os.makedirs(save_model_dir, exist_ok=True)
#         logger.info(f"Ensured directory for model logs/checkpoints exists: {save_model_dir}")
#         return opt
#     except FileNotFoundError:
#         logger.error(f"Config file not found: {file_path}")
#         raise
#     except yaml.YAMLError as e:
#         logger.error(f"Error parsing YAML config file {file_path}: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"An unexpected error occurred in get_config: {e}")
#         raise


# def finetune(data):
#     logger.info("Starting finetune data preparation and training process.")

#     # --- FIXED: Set a random seed for reproducible train/val splits ---
#     random_seed = 42
#     random.seed(random_seed)
#     mysql_db = next(get_db())
#     logger.info(f"Random seed set to {random_seed} for reproducible data splits.")

#     MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
#     logger.info(f"Manuscripts base path: {MANUSCRIPTS_PATH}")

#     # Extracting parameters from the input data
#     # Assuming data is a list of dicts, and we take the first item
#     if not data or not isinstance(data, list) or not data[0]:
#         logger.error("Input 'data' is empty or incorrectly formatted.")
#         raise ValueError("Input 'data' must be a non-empty list of dictionaries.")

#     manuscript_name = data[0].get("manuscript_name")
#     annotations = data[0].get("annotations")
#     selected_model = data[0].get("selected_model")
#     model_name = data[0].get("model_name", f"{manuscript_name}.pth")

#     if not all([manuscript_name, annotations, selected_model]):
#         logger.error(f"Missing essential parameters in input data: manuscript_name={manuscript_name}, annotations={bool(annotations)}, selected_model={selected_model}")
#         raise ValueError("Missing essential parameters in input data.")

#     logger.info(f"Input details - Manuscript: '{manuscript_name}', Selected Model: '{selected_model}', Output Model Name: '{model_name}'")
#     # logger.debug(f"Raw annotations data (can be large): {annotations}") # Keep commented for production, uncomment for deep debugging

#     opt = get_config(
#         os.path.join("annotator", "finetune", "config_files", "config.yml"),
#         manuscript_name,
#         selected_model,
#         model_name,
#     )
#     logger.info("Configuration loaded successfully.")

#     TEMP_FOLDER = "temp"
#     TRAIN_FOLDER = os.path.join(TEMP_FOLDER, "train")
#     VAL_FOLDER = os.path.join(TEMP_FOLDER, "val")
#     TRAIN_CSV_FILE = os.path.join(TRAIN_FOLDER, "labels.csv")
#     VAL_CSV_FILE = os.path.join(VAL_FOLDER, "labels.csv")
#     logger.debug(f"Temporary folders: TRAIN={TRAIN_FOLDER}, VAL={VAL_FOLDER}")

#     # --- FIXED: Ensure a clean slate by removing the temp folder if it exists ---
#     if os.path.exists(TEMP_FOLDER):
#         logger.warning(f"Existing temporary folder '{TEMP_FOLDER}' found. Deleting to ensure a clean slate.")
#         shutil.rmtree(TEMP_FOLDER)
#     else:
#         logger.info(f"Temporary folder '{TEMP_FOLDER}' does not exist, proceeding to create.")

#     # Create the necessary directories
#     os.makedirs(TRAIN_FOLDER, exist_ok=True)
#     os.makedirs(VAL_FOLDER, exist_ok=True)
#     logger.info(f"Created temporary training data directories: {TRAIN_FOLDER} and {VAL_FOLDER}")
    
#     # --- IMPROVEMENT: Collect all data points first before splitting ---
#     all_data_points = []
#     log_entries = [] # Collect log entries to add to DB in a batch
#     skipped_images_count = 0

#     logger.info("Collecting all annotation data points and creating DB log entries.")
#     for page in annotations:
#         logger.debug(f"Processing page: {page}")
#         for line in annotations[page]:
#             logger.debug(f"Processing line: {line} on page: {page}")
#             ground_truth = annotations[page][line]["ground_truth"]
#             image_path = os.path.join(
#                 MANUSCRIPTS_PATH, manuscript_name, "lines", page, line + ".jpg"
#             )
            
#             # --- FIXED: Create a unique filename to prevent collisions ---
#             unique_filename = f"{page}_{line}.jpg"

#             # Create log entry, but add to a list to commit later
#             # This is more efficient for the database
#             log_entry = UserAnnotationLog(
#                 manuscript_name=manuscript_name,
#                 page=page,
#                 line=line,
#                 ground_truth=ground_truth,
#                 levenshtein_distance=annotations[page][line]["levenshtein_distance"],
#                 image_path=image_path,
#                 timestamp=datetime.now(),
#             )
#             log_entries.append(log_entry)

            
#             if os.path.exists(image_path):
#                  all_data_points.append({
#                     "image_path": image_path,
#                     "unique_filename": unique_filename,
#                     "ground_truth": ground_truth
#                 })
#                  logger.debug(f"Collected data for {unique_filename} with ground truth: '{ground_truth}'")
#             else:
#                  logger.warning(f"Image not found, skipping annotation for: {image_path}")
#                  skipped_images_count += 1

#     db.session.add_all(log_entries)
#     logger.info(f"Prepared {len(log_entries)} database log entries. Skipped {skipped_images_count} image files due to not being found.")
#     for page in annotations:
#         for line in annotations[page]:
#             ground_truth = annotations[page][line]["ground_truth"]
#             levenshtein_distance = annotations[page][line]["levenshtein_distance"]
#             image_path = os.path.join(
#                 MANUSCRIPTS_PATH, manuscript_name, "lines", page, line + ".jpg"
#             )
#             unique_filename = f"{page}_{line}.jpg"

#             if not os.path.exists(image_path):
#                 logger.warning(f"Image not found, skipping annotation for: {image_path}")
#                 skipped_images_count += 1
#                 continue

#             # Check if log exists for this manuscript/page/line/model
#             existing_log = mysql_db.query(AnnotationLog).filter_by(
#                 manuscript_name=manuscript_name,
#                 page=page,
#                 line=line,
#                 model_selected=selected_model
#             ).first()

#             if existing_log:
#                 existing_log.ground_truth = ground_truth
#                 existing_log.levenshtein_distance = levenshtein_distance
#                 logger.debug(f"Updated existing AnnotationLog for {unique_filename}")
#             else:
#                 log_entry = AnnotationLog(
#                     predicted_label="",
#                     confidence_score=0.0,
#                     manuscript_name=manuscript_name,
#                     ground_truth=ground_truth,
#                     levenshtein_distance=levenshtein_distance,
#                     page=page,
#                     line=line,
#                     image_path=image_path,
#                     model_selected=selected_model,
#                     timestamp=datetime.now()
#                 )
#                 mysql_db.add(log_entry)

#             # Add to data points for training/validation
#             all_data_points.append({
#                 "image_path": image_path,
#                 "unique_filename": unique_filename,
#                 "ground_truth": ground_truth
#             })

#     # Commit AnnotationLog entries
#     try:
#         mysql_db.commit()
#         logger.info(f"AnnotationLog entries committed. Skipped {skipped_images_count} missing images.")
#     except SQLAlchemyError as e:
#         mysql_db.rollback()
#         logger.error(f"Failed to commit AnnotationLog entries: {e}")
#         raise
#     if not all_data_points:
#         logger.error("No valid image-annotation pairs were collected. Cannot proceed with fine-tuning.")
#         db.session.commit() # Commit any partial logs
#         shutil.rmtree(TEMP_FOLDER) # Clean up temp folder even on error
#         raise ValueError("No valid data points for fine-tuning.")


#     logger.info(f"Total {len(all_data_points)} data points collected before splitting.")

#     # Shuffle the dataset before splitting for better distribution
#     random.shuffle(all_data_points)
#     logger.info("Data points shuffled for unbiased splitting.")
    
#     # Split the data into training and validation sets (80/20)
#     split_index = int(0.8 * len(all_data_points))
#     train_data = all_data_points[:split_index]
#     val_data = all_data_points[split_index:]

#     logger.info(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples.")

#     # --- RESTRUCTURED: Process training and validation data separately ---
    
#     # Process training data
#     logger.info(f"Writing training data to {TRAIN_CSV_FILE} and copying images to {TRAIN_FOLDER}")
#     try:
#         with open(TRAIN_CSV_FILE, mode="w", encoding="utf-8", newline="") as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerow(["filename", "words"])
#             for i, item in enumerate(train_data):
#                 shutil.copy(item["image_path"], os.path.join(TRAIN_FOLDER, item["unique_filename"]))
#                 csvwriter.writerow([item["unique_filename"], item["ground_truth"]])
#                 if (i + 1) % 100 == 0:
#                     logger.debug(f"Copied {i+1}/{len(train_data)} training images.")
#         logger.info(f"Successfully wrote {len(train_data)} training samples to CSV and copied images.")
#     except Exception as e:
#         logger.error(f"Error processing training data: {e}")
#         db.session.rollback() # Rollback DB changes if data creation fails
#         raise

            
#     # Process validation data
#     logger.info(f"Writing validation data to {VAL_CSV_FILE} and copying images to {VAL_FOLDER}")
#     try:
#         with open(VAL_CSV_FILE, mode="w", encoding="utf-8", newline="") as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerow(["filename", "words"])
#             for i, item in enumerate(val_data):
#                 shutil.copy(item["image_path"], os.path.join(VAL_FOLDER, item["unique_filename"]))
#                 csvwriter.writerow([item["unique_filename"], item["ground_truth"]])
#                 if (i + 1) % 50 == 0:
#                     logger.debug(f"Copied {i+1}/{len(val_data)} validation images.")
#         logger.info(f"Successfully wrote {len(val_data)} validation samples to CSV and copied images.")
#     except Exception as e:
#         logger.error(f"Error processing validation data: {e}")
#         db.session.rollback() # Rollback DB changes if data creation fails
#         raise

#     # Commit all collected UserAnnotationLog entries to the database
#     try:
#         db.session.commit()
#         logger.info("All UserAnnotationLog entries committed to the database.")
#     except Exception as e:
#         logger.error(f"Failed to commit UserAnnotationLog entries to database: {e}")
#         db.session.rollback()
#         raise

#     logger.info("Data preparation complete. Initiating model training.")
#     try:
#         train(opt, manuscript_name, amp=False)
#         logger.info("Model training function `train()` executed successfully.")
#     except Exception as e:
#         logger.error(f"An error occurred during model training: {e}")
#         # Depending on requirements, you might want to rollback more DB changes or clean up models here
#         raise

#     # Clean up the temporary folder after training is complete
#     try:
#         shutil.rmtree(TEMP_FOLDER)
#         logger.info(f"Cleaned up temporary folder: '{TEMP_FOLDER}'.")
#     except Exception as e:
#         logger.error(f"Failed to remove temporary folder '{TEMP_FOLDER}': {e}")
#         # This error is not critical for the overall process but should be logged.

#     logger.info("Finetune process finished successfully.")
    


import sys
import os
import csv
import shutil
import random
import yaml
import pandas as pd
import logging  # Import the logging module

from datetime import datetime
from flask import current_app

from annotator.finetune.utils import AttrDict
from annotator.finetune.train import train
from database.connection import get_db
from model.manuscriptmodel import AnnotationLog
from sqlalchemy.exc import SQLAlchemyError


# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure stdout/stderr are configured for UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def get_config(file_path, manuscript_name, selected_model, model_name):
    logger.info(f"Loading configuration from: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as stream:
            opt = yaml.safe_load(stream)
        opt = AttrDict(opt)
        opt.character = opt.number + opt.symbol + opt.lang_char
        opt.manuscript_name = manuscript_name
        opt.saved_model = os.path.join(
            os.path.join(current_app.config['DATA_PATH']), "models", "recognition", selected_model,
        )
        opt.model_name = model_name

        logger.info(
            f"Config loaded for manuscript: '{manuscript_name}' with model: '{selected_model}' (output name: '{model_name}')"
        )
        logger.debug(f"Full config opt object: {opt}")

        save_model_dir = f"./saved_models/{opt.model_name}"
        os.makedirs(save_model_dir, exist_ok=True)
        logger.info(f"Ensured directory for model logs/checkpoints exists: {save_model_dir}")
        return opt
    except FileNotFoundError:
        logger.error(f"Config file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_config: {e}")
        raise


def finetune(data):
    logger.info("Starting finetune data preparation and training process.")

    # Set random seed for reproducible train/val splits
    random_seed = 42
    random.seed(random_seed)
    mysql_db = next(get_db())
    logger.info(f"Random seed set to {random_seed} for reproducible data splits.")

    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    logger.info(f"Manuscripts base path: {MANUSCRIPTS_PATH}")

    if not data or not isinstance(data, list) or not data[0]:
        logger.error("Input 'data' is empty or incorrectly formatted.")
        raise ValueError("Input 'data' must be a non-empty list of dictionaries.")

    manuscript_name = data[0].get("manuscript_name")
    annotations = data[0].get("annotations")
    selected_model = data[0].get("selected_model")
    model_name = data[0].get("model_name", f"{manuscript_name}.pth")

    if not all([manuscript_name, annotations, selected_model]):
        logger.error(
            f"Missing essential parameters in input data: manuscript_name={manuscript_name}, annotations={bool(annotations)}, selected_model={selected_model}"
        )
        raise ValueError("Missing essential parameters in input data.")

    logger.info(
        f"Input details - Manuscript: '{manuscript_name}', Selected Model: '{selected_model}', Output Model Name: '{model_name}'"
    )

    opt = get_config(
        os.path.join("annotator", "finetune", "config_files", "config.yml"),
        manuscript_name,
        selected_model,
        model_name,
    )
    logger.info("Configuration loaded successfully.")

    TEMP_FOLDER = "temp"
    TRAIN_FOLDER = os.path.join(TEMP_FOLDER, "train")
    VAL_FOLDER = os.path.join(TEMP_FOLDER, "val")
    TRAIN_CSV_FILE = os.path.join(TRAIN_FOLDER, "labels.csv")
    VAL_CSV_FILE = os.path.join(VAL_FOLDER, "labels.csv")

    if os.path.exists(TEMP_FOLDER):
        logger.warning(f"Existing temporary folder '{TEMP_FOLDER}' found. Deleting to ensure a clean slate.")
        shutil.rmtree(TEMP_FOLDER)
    else:
        logger.info(f"Temporary folder '{TEMP_FOLDER}' does not exist, proceeding to create.")

    os.makedirs(TRAIN_FOLDER, exist_ok=True)
    os.makedirs(VAL_FOLDER, exist_ok=True)
    logger.info(f"Created temporary training data directories: {TRAIN_FOLDER} and {VAL_FOLDER}")

    all_data_points = []
    skipped_images_count = 0

    logger.info("Collecting all annotation data points.")
    for page in annotations:
        logger.debug(f"Processing page: {page}")
        for line in annotations[page]:
            logger.debug(f"Processing line: {line} on page: {page}")
            ground_truth = annotations[page][line]["ground_truth"]
            image_path = os.path.join(
                MANUSCRIPTS_PATH, manuscript_name, "lines", page, line + ".jpg"
            )
            unique_filename = f"{page}_{line}.jpg"

            if os.path.exists(image_path):
                all_data_points.append({
                    "image_path": image_path,
                    "unique_filename": unique_filename,
                    "ground_truth": ground_truth
                })
                logger.debug(f"Collected data for {unique_filename} with ground truth: '{ground_truth}'")
            else:
                logger.warning(f"Image not found, skipping annotation for: {image_path}")
                skipped_images_count += 1

    logger.info(f"Prepared {len(all_data_points)} data points. Skipped {skipped_images_count} image files.")

    for page in annotations:
        for line in annotations[page]:
            ground_truth = annotations[page][line]["ground_truth"]
            levenshtein_distance = annotations[page][line]["levenshtein_distance"]
            image_path = os.path.join(
                MANUSCRIPTS_PATH, manuscript_name, "lines", page, line + ".jpg"
            )
            unique_filename = f"{page}_{line}.jpg"

            if not os.path.exists(image_path):
                logger.warning(f"Image not found, skipping annotation for: {image_path}")
                skipped_images_count += 1
                continue

            existing_log = mysql_db.query(AnnotationLog).filter_by(
                manuscript_name=manuscript_name,
                page=page,
                line=line,
                model_selected=selected_model
            ).first()

            if existing_log:
                existing_log.ground_truth = ground_truth
                existing_log.levenshtein_distance = levenshtein_distance
                logger.debug(f"Updated existing AnnotationLog for {unique_filename}")
            else:
                log_entry = AnnotationLog(
                    predicted_label="",
                    confidence_score=0.0,
                    manuscript_name=manuscript_name,
                    ground_truth=ground_truth,
                    levenshtein_distance=levenshtein_distance,
                    page=page,
                    line=line,
                    image_path=image_path,
                    model_selected=selected_model,
                    timestamp=datetime.now()
                )
                mysql_db.add(log_entry)

            all_data_points.append({
                "image_path": image_path,
                "unique_filename": unique_filename,
                "ground_truth": ground_truth
            })

    try:
        mysql_db.commit()
        logger.info(f"AnnotationLog entries committed. Skipped {skipped_images_count} missing images.")
    except SQLAlchemyError as e:
        mysql_db.rollback()
        logger.error(f"Failed to commit AnnotationLog entries: {e}")
        raise

    if not all_data_points:
        logger.error("No valid image-annotation pairs were collected. Cannot proceed with fine-tuning.")
        shutil.rmtree(TEMP_FOLDER)
        raise ValueError("No valid data points for fine-tuning.")

    logger.info(f"Total {len(all_data_points)} data points collected before splitting.")

    random.shuffle(all_data_points)
    logger.info("Data points shuffled for unbiased splitting.")

    split_index = int(0.8 * len(all_data_points))
    train_data = all_data_points[:split_index]
    val_data = all_data_points[split_index:]

    logger.info(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples.")

    # Write training data
    logger.info(f"Writing training data to {TRAIN_CSV_FILE} and copying images to {TRAIN_FOLDER}")
    try:
        with open(TRAIN_CSV_FILE, mode="w", encoding="utf-8", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["filename", "words"])
            for i, item in enumerate(train_data):
                shutil.copy(item["image_path"], os.path.join(TRAIN_FOLDER, item["unique_filename"]))
                csvwriter.writerow([item["unique_filename"], item["ground_truth"]])
                if (i + 1) % 100 == 0:
                    logger.debug(f"Copied {i+1}/{len(train_data)} training images.")
        logger.info(f"Successfully wrote {len(train_data)} training samples.")
    except Exception as e:
        logger.error(f"Error processing training data: {e}")
        raise

    # Write validation data
    logger.info(f"Writing validation data to {VAL_CSV_FILE} and copying images to {VAL_FOLDER}")
    try:
        with open(VAL_CSV_FILE, mode="w", encoding="utf-8", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["filename", "words"])
            for i, item in enumerate(val_data):
                shutil.copy(item["image_path"], os.path.join(VAL_FOLDER, item["unique_filename"]))
                csvwriter.writerow([item["unique_filename"], item["ground_truth"]])
                if (i + 1) % 50 == 0:
                    logger.debug(f"Copied {i+1}/{len(val_data)} validation images.")
        logger.info(f"Successfully wrote {len(val_data)} validation samples.")
    except Exception as e:
        logger.error(f"Error processing validation data: {e}")
        raise

    logger.info("Data preparation complete. Initiating model training.")
    try:
        train(opt, manuscript_name, amp=False)
        logger.info("Model training function `train()` executed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise

    try:
        shutil.rmtree(TEMP_FOLDER)
        logger.info(f"Cleaned up temporary folder: '{TEMP_FOLDER}'.")
    except Exception as e:
        logger.error(f"Failed to remove temporary folder '{TEMP_FOLDER}': {e}")

    logger.info("Finetune process finished successfully.")
