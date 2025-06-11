# OCR Tool for Historical Devanagari Manuscripts
[Kartik Chincholikar ](https://kartikchincholikar.github.io/), [Shagun Dwivedi ](https://shagundwivedi.github.io/), [Bharath Valaboju](https://Bharath314.github.io/), [Kaushik Gopalan](https://www.flame.edu.in/faculty/kaushik-gopalan)
<!-- **[Paper](https://arxiv.org/abs/2502.12534), [Project Page](https://theialab.github.io/noksr/)** -->
<!-- ![noksr](assets/Teaser.png) -->

![Demo](demo.gif)

*Step 1: Automatically Segment Text Line Images from Document, with the ability to manually ADD or DELETE edges for tricky layouts. Step 2: Recognize the text content from the Text Line Images, make corrections, and fine tune the IMG2TEXT model*

Digitizing text from historical manuscripts yields historians multiple benefits. The digitization process consists of three steps: text-line-image segmentation, text recognition from the text-line-images (and post-correction). 

This tool enables segmenting text-line-images from pages with diverse layouts. It represents text-lines as graphs, with characters as the nodes, and with edges connecting each character of a text-line to it's previous and next neighbour. In other words, we use nodes and edges as units of comparison and data collection instead of dense pixel-level metrics. This enables easier layout annotation, and improved performance compared to existing methods (as tested on a set of 15 pages with layouts of varying complexity, ranging from simple single-column and double-column layouts to layouts with pictures, footnotes, tables, interlinear writing, marginalia, text bleeding, staining, coloring, and irregular font sizes)

To recognise text content from the segmented text-line-images, we use a pre-trained text recognition model for the Devanāgarī script. The tools enables fine-tuning of the pre-trained model on specific manuscripts, which results in the model's predictions getting progressively better with more annotated data, thus also making the subsequent annotation easier - similar to active learning.

Contact kartik.niszoig at gmail for questions, comments and reporting bugs.

## News    

- [2025/05/30] Code Released!

## Environment Setup
The code is tested on Windows 11 (x64) machine with NVIDIA GeForce RTX 4050 Laptop GPU with CUDA 12.8 Driver. 

```
# Download/Clone this repository
git clone https://github.com/flame-cai/win64-local-ocr-tool.git

# go to folder win64-local-ocr-tool
cd win64-local-ocr-tool
```

The application uses two AI models: [CRAFT](https://github.com/clovaai/CRAFT-pytorch) and [EasyOCR's](https://github.com/JaidedAI) Devanagari pretrained model. CRAFT detects the locations of the characters in a page, which is used to crop out text-line-images from pages with diverse layouts. The Devanagari pretrained model is then used to detect the text-content from the cropped text-line-images, and can also be fine-tuned for a specific manuscript. 

- Download craft_mlt_25k.pth from [here](https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth?download=true). Put this file in the `backend/instance/models/segmentation/` folder. 

- Download devanagari.pth from [here](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip). Make sure to unzip the devanagari.zip file to get devanagari.pth file. Put this file in the `backend/instance/models/recognition/` folder. 


### Setup the backend
Please follow the following steps to create the backend conda environment:
```
# open terminal (or miniconda prompt) and go to the backend folder
cd backend

# create the conda environment
conda env create -f environment.yml

# activate the conda environment named 'ocr-tool'
conda activate ocr-tool

# run the backend
flask run --debug

# OR run the backend using:  
python app.py
```

### Setup the frontend
Install [Node.js](https://nodejs.org/en) if not installed.
```
# open a new terminal, and go to frontend folder
cd frontend

# Install the node packages using
npm install

# Run the development server using 
npm run dev
```

## TODO

- [ ] record location of the segmented line on map as meta data..
- [ ] Integrate V2 English model
- [ ] Multi-layerd line segmentation based on font size
- [ ] Cuda profiling 
- [ ] UX zoom - maintain aspect ratio
- [ ] Per line iterative finetuning
- [ ] benefits of "overdoing" finetuning: [Ref1](https://arxiv.org/pdf/2408.04809) [Ref2](https://imtiazhumayun.github.io/grokking/)
- [ ] integrate ByT5-Sanskrit with this tool and auto-correct the OCR output
- [ ] finetune [ByT5-Sanskrit](https://huggingface.co/chronbmm/sanskrit-byt5-ocr-postcorrection) using [reinforcement learning](https://arxiv.org/abs/2501.17161)

 
