import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import csv
import shutil
import random
import yaml
import pandas as pd

from datetime import datetime
from flask import current_app

from annotator.finetune.utils import AttrDict
from annotator.finetune.train import train
from annotator.models import db, UserAnnotationLog


def get_config(file_path, manuscript_name, selected_model, model_name):
    with open(file_path, "r", encoding="utf-8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt.character = opt.number + opt.symbol + opt.lang_char
    opt.manuscript_name = manuscript_name
    opt.saved_model = os.path.join(
        os.path.join(current_app.config['DATA_PATH']), "models", "recognition", selected_model,
    )
    opt.model_name = model_name
    # This creates place for logfile. Need to keep this in mind when transitioning logs from .txt to database
    os.makedirs(f"./saved_models/{opt.model_name}", exist_ok=True)
    return opt


def finetune(data):
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')

    manuscript_name = data[0]["manuscript_name"]
    annotations = data[0]["annotations"]
    selected_model = data[0]["selected_model"]
    model_name = data[0].get("model_name", f"{manuscript_name}.pth")

    opt = get_config(
        os.path.join("backend","annotator", "finetune", "config_files", "config.yml"),
        manuscript_name,
        selected_model,
        model_name,
    )

    TEMP_FOLDER = "temp"
    TRAIN_FOLDER = os.path.join(TEMP_FOLDER, "train")
    VAL_FOLDER = os.path.join(TEMP_FOLDER, "val")
    TRAIN_CSV_FILE = os.path.join(TRAIN_FOLDER, "labels.csv")
    VAL_CSV_FILE = os.path.join(VAL_FOLDER, "labels.csv")

    # Ensure the temp folder exists
    os.makedirs(TRAIN_FOLDER, exist_ok=True)
    os.makedirs(VAL_FOLDER, exist_ok=True)

    # Initialize the CSV files with headers if they don't exist
    for csv_file in [TRAIN_CSV_FILE, VAL_CSV_FILE]:
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", encoding="utf-8", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["filename", "words"])

    for page in annotations:
        for line in annotations[page]:
            ground_truth = annotations[page][line]["ground_truth"]
            image_path = os.path.join(
                MANUSCRIPTS_PATH, manuscript_name, "lines", page, line + ".jpg"
            )
            filename = os.path.basename(image_path)

            # Create log entry
            log_entry = UserAnnotationLog(
                manuscript_name=manuscript_name,
                page=page,
                line=line,
                ground_truth=ground_truth,
                levenshtein_distance=annotations[page][line]["levenshtein_distance"],
                image_path=image_path,
                timestamp=datetime.now(),
            )
            db.session.add(log_entry)

            # Randomly assign to train or val (80% train, 20% val)
            if random.random() < 0.8:
                target_folder = TRAIN_FOLDER
                target_csv = TRAIN_CSV_FILE
            else:
                target_folder = VAL_FOLDER
                target_csv = VAL_CSV_FILE

            # Copy the image to the appropriate folder
            try:
                shutil.copy(image_path, target_folder)
            except FileNotFoundError:
                print(f"Image not found: {image_path}")

            # Append to the appropriate CSV file
            with open(target_csv, mode="a", encoding="utf-8", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([filename, ground_truth])
    db.session.commit()

    train(opt, manuscript_name, amp=False)

    shutil.rmtree("temp")
