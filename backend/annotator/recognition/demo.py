import string
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from annotator.recognition.utils import CTCLabelConverter, AttnLabelConverter
from annotator.recognition.dataset import RawDataset, AlignCollate
from annotator.recognition.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OCRConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def recognise_lines(
    image_folder,
    saved_model,
    transformation,
    feature_extraction,
    sequence_modeling,
    prediction,
    batch_size=192,
    workers=4,
    batch_max_length=25,
    imgH=32,
    imgW=100,
    rgb=False,
    character=None,
    sensitive=False,
    pad=False,
    num_fiducial=20,
    input_channel=1,
    output_channel=512,
    hidden_size=256,
):
    """
    Recognise text lines from images in the specified folder using the specified model.

    Parameters:
        image_folder (str): Path to the folder containing images.
        saved_model (str): Path to the pretrained model.
        transformation (str): Transformation stage. Options: None, TPS.
        feature_extraction (str): Feature extraction stage. Options: VGG, RCNN, ResNet.
        sequence_modeling (str): Sequence modeling stage. Options: None, BiLSTM.
        prediction (str): Prediction stage. Options: CTC, Attn.
        batch_size (int): Batch size for processing images.
        workers (int): Number of workers for data loading.
        batch_max_length (int): Maximum label length.
        imgH (int): Height of input images.
        imgW (int): Width of input images.
        rgb (bool): Whether to use RGB input.
        character (str): Character set for labels.
        sensitive (bool): Use sensitive character mode.
        pad (bool): Whether to pad resized images to maintain aspect ratio.
        num_fiducial (int): Number of fiducial points for TPS-STN.
        input_channel (int): Number of input channels for the feature extractor.
        output_channel (int): Number of output channels for the feature extractor.
        hidden_size (int): Size of the LSTM hidden state.

    Returns:
        results (list): List of dictionaries containing image paths, predicted labels, and confidence scores.
    """

    # Configure the character set
    if sensitive:
        character = string.printable[:-6] if character is None else character
    elif character is None:
        character = "0123456789abcdefghijklmnopqrstuvwxyz"

    # Initialize label converter
    if "CTC" in prediction:
        converter = CTCLabelConverter(character)
    else:
        converter = AttnLabelConverter(character)
    num_class = len(converter.character)

    # Set input channel for RGB images
    input_channel = 3 if rgb else input_channel

    opt = OCRConfig(
        image_folder=image_folder,
        saved_model=saved_model,
        Transformation=transformation,
        FeatureExtraction=feature_extraction,
        SequenceModeling=sequence_modeling,
        Prediction=prediction,
        batch_size=batch_size,
        workers=workers,
        batch_max_length=batch_max_length,
        imgH=imgH,
        imgW=imgW,
        rgb=rgb,
        character=character,
        sensitive=sensitive,
        PAD=pad,
        num_fiducial=num_fiducial,
        input_channel=input_channel,
        output_channel=output_channel,
        hidden_size=hidden_size,
        num_class=num_class,
    )

    # Load the model
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    print(f"Loading pretrained model from {saved_model}")
    model.load_state_dict(torch.load(saved_model, map_location=device))

    # Prepare data loader
    AlignCollate_demo = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=pad)
    demo_data = RawDataset(root=image_folder, opt=opt)  # Use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=AlignCollate_demo,
        pin_memory=True,
    )

    # Perform prediction
    model.eval()
    results = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(
                device
            )
            text_for_pred = (
                torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)
            )

            if "CTC" in prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
                del preds_size, preds_index
            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_name, pred, pred_max_prob in zip(
                image_path_list, preds_str, preds_max_prob
            ):
                if "Attn" in prediction:
                    pred_EOS = pred.find("[s]")
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                results.append(
                    {
                        "image_path": img_name,
                        "predicted_label": pred,
                        "confidence_score": confidence_score.item(),
                    }
                )
            del image, length_for_pred, text_for_pred, preds, preds_prob, preds_max_prob
            if 'preds_size' in locals(): del preds_size
            if 'preds_index' in locals(): del preds_index
            torch.cuda.empty_cache()

    
    # clear GPU memory
    del model
    del demo_loader, AlignCollate_demo, demo_data
    torch.cuda.empty_cache()

    return results
