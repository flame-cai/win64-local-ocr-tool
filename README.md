# OCR Tool for Historical Devanagari Manuscripts
[Kartik Chincholikar ](https://kartikchincholikar.github.io/), [Shagun Dwivedi ](https://shagundwivedi.github.io/), [Bharath Valaboju](https://Bharath314.github.io/)
<!-- **[Paper](https://arxiv.org/abs/2502.12534), [Project Page](https://theialab.github.io/noksr/)** -->
<!-- ![noksr](assets/Teaser.png) -->

Digitizing text from historical manuscripts yields historians multiple benefits. The digitization process consists of three steps: text-line-image segmentation, text recognition from the text-line-images (and post-correction). 

This tool enables segmenting text-line-images from pages with diverse layouts - representing text-lines as graphs, with characters as the nodes, and with edges connecting characters to their previous and next characters on that text-line. We use this graph-based representation to annotate data and evaluate text-line segmentation predictions, using nodes and edges as units of comparison instead of pixel-level metrics. We highlight the advantages of this graph-based representation and propose a simple semi-automatic text-line segmentation method, which uses deep learning to locate the characters of the page, and then uses the aforementioned graph representation to devise a simple algorithm that connects characters of a text-line. We test this proposed algorithm on a set of 15 pages with layouts of varying complexity, ranging from simple single-column and double-column layouts to layouts with pictures, footnotes, marginalia, tables, and irregular font sizes. We find that the proposed algorithm performs reasonably well on metrics such as Adjusted Rand Index, V-measure and Graph Edit Distance.

To recognise text content from the segmented text-line-images, we use a pre-trained text recognition model for the Devanāgarī script. The tools enables fine-tuning of the pre-trained model on specific manuscripts, which results in the model's predictions getting progressively better with more annotated data, thus also making the subsequent annotation easier - similar to active learning.

Contact kartik.niszoig at gmail for questions, comments and reporting bugs.

## News    

- [2025/05/30] Code Released!

## Environment Setup
The code is tested on Windows 11 (x64) machine with NVIDIA GeForce RTX 4050 Laptop GPU with CUDA 12.8 Driver. We used Python 3.11 with cuda-12.1.1 Runtime. Please follow the following steps to create the conda environment:

```
# Download/Clone this repository
git clone https://github.com/flame-cai/win64-local-ocr-tool.git

# go to folder win64-local-ocr-tool
cd win64-local-ocr-tool
```

The application uses two AI model: [CRAFT](https://github.com/clovaai/CRAFT-pytorch) and [EasyOCR's](https://github.com/JaidedAI) Devanagari pretrained model. CRAFT detext the locations of the characters in a page, which is used to crop out text-line-images from pages with diverse layouts. The Devanagari pretrained model is then used to detect the text-content from the cropped text-line-images, and can also be fine-tuned for a specific manuscript. 

- Download craft_mlt_25k.pth from [here](https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth?download=true). Put this file in the `backend/instance/models/segmentation` folder.

- Download devanagari.pth from [here](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip). Put this file in the `backend/instance/models/recognition` folder.


### Setup the backend
```
# go to the backend folder
cd backend

# create the conda environment 'san-ocr-tool'
conda env create -f environment.yml

# activate the conda environment named 'san-ocr-tool'
conda activate san-ocr-tool

# run the backend
flask run --debug
```

### Setup the frontend
```
# Install node.js

# open a new terminal, and go to frontend folder
cd frontend

# Install the node packages using
npm install

# Run the development server using 
npm run dev
```

## TODO
- integrate ByT5-Sanskrit with this tool and auto-correct the OCR output
- finetune [ByT5-Sanskrit](https://huggingface.co/chronbmm/sanskrit-byt5-ocr-postcorrection) using [reinforcement learning](https://arxiv.org/abs/2501.17161)

 
