import os
import subprocess
import torch

from datetime import datetime
from flask import current_app

from annotator.recognition.demo import recognise_lines
from annotator.models import db, RecognitionLog

def get_filename_without_extension(file_path):
    """
    Extracts the filename without extension from a given file path.

    :param file_path: str - The full file path.
    :return: str - The filename without the extension.
    """
    # Extract the base name of the file
    base_name = os.path.basename(file_path)
    # Remove the file extension
    file_name, _ = os.path.splitext(base_name)
    return file_name

def get_subfolders(folder_path):
    return [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

def recognise_characters(folder_path, model, manuscript_name):
    lines_of_all_pages = {}
    lines_folder_path = os.path.join(folder_path, "lines")
    page_subfolders = get_subfolders(lines_folder_path)
    for page_subfolder in page_subfolders:
        lines_of_one_page = recognise_lines(
            image_folder=os.path.join(lines_folder_path, page_subfolder),
            saved_model=os.path.join(current_app.config['DATA_PATH'], 'models', 'recognition', model),
            transformation=None,
            feature_extraction="ResNet",
            sequence_modeling="BiLSTM",
            prediction="CTC",
            workers=0,
            batch_max_length=250,
            imgH=50,
            imgW=2000,
            pad=True,
            character="""`0123456789~!@#$%^&*()-_+=[]\\{}|;':",./<>? abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰""",
            hidden_size=512,
            output_channel=512,
        )
        for line in lines_of_one_page:
            line["manuscript_name"] = manuscript_name
            line["selected_model"] = model
            line["page"] = page_subfolder
            line["line"] = get_filename_without_extension(line["image_path"])

            # Add model name to Log
            log_entry = RecognitionLog(
                image_path=line["image_path"],
                predicted_label=line["predicted_label"],
                confidence_score=line["confidence_score"],
                manuscript_name=manuscript_name,
                page=page_subfolder,
                line=line["line"],
                timestamp=datetime.now()
            )
            db.session.add(log_entry)
        db.session.commit()
        lines_of_all_pages[page_subfolder] = lines_of_one_page
    
    # clear GPU memory
    del lines_of_one_page
    torch.cuda.empty_cache()
    
    return lines_of_all_pages