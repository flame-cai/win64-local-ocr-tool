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


# In annotator/recognition/recognition.py (or wherever recognise_characters is)
# Make sure to import: os, current_app, datetime, db, RecognitionLog, torch, get_filename_without_extension
# And from annotator.recognition.demo import recognise_lines

def recognise_single_page_characters(manuscript_folder_path, model_name, manuscript_name, page_to_process):
    """
    Recognises characters for a single specified page of a manuscript.
    Returns a dictionary of line data for that page.
    """
    lines_data_for_page = {}
    # Path to the specific page's line images
    page_lines_folder = os.path.join(manuscript_folder_path, "lines", page_to_process)

    if not os.path.isdir(page_lines_folder):
        current_app.logger.warning(f"Lines folder not found for page: {page_lines_folder}")
        return {} # Return empty if no lines found for the page

    # Recognise lines in the specified page_lines_folder
    # Note: ensure recognise_lines correctly processes images in this folder
    recognized_lines_list = recognise_lines(
        image_folder=page_lines_folder,
        saved_model=os.path.join(current_app.config['DATA_PATH'], 'models', 'recognition', model_name),
        transformation=None, # Add other params as they are in recognise_characters
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

    for line_info in recognized_lines_list:
        # 'image_path' from recognise_lines is the full fs path to the line image.
        # We need the line name (filename without ext) for the store and URL construction.
        line_name = get_filename_without_extension(line_info["image_path"])

        lines_data_for_page[line_name] = {
            "predicted_label": line_info["predicted_label"],
            # This 'image_path' will be used by frontend to construct the request to /line-images endpoint.
            # So, it should be the line identifier (filename without extension).
            "image_path": line_name,
            "confidence_score": line_info["confidence_score"],
            # Optionally include these if frontend store needs them per line directly,
            # though typically they are known from manuscript_name and modelName in store.
            # "manuscript_name": manuscript_name,
            # "selected_model": model_name
        }

        # Log to DB
        log_entry = RecognitionLog(
            image_path=line_info["image_path"], # Log the actual filesystem path
            predicted_label=line_info["predicted_label"],
            confidence_score=line_info["confidence_score"],
            manuscript_name=manuscript_name,
            page=page_to_process,
            line=line_name, # Log the line name
            timestamp=datetime.now()
        )
        db.session.add(log_entry)
    
    db.session.commit()
    # GPU memory clear can be done after this call in the route handler
    return lines_data_for_page