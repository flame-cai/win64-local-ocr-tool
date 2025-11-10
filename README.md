# Semi-Automatic Layout Analysis and Handwritten Text Recognition Tool for Historical Devanagari Manuscripts
[Kartik Chincholikar ](https://kartikchincholikar.github.io/), [Shagun Dwivedi ](https://shagundwivedi.github.io/), [Bharath Valaboju](https://Bharath314.github.io/), [Kaushik Gopalan](https://www.linkedin.com/in/kaushik-gopalan-b6533624/?originalSubdomain=in), [Tarinee Awasthi](https://www.linkedin.com/in/tarinee-awasthi-89883a244/), [Vinod Vidwans](https://www.linkedin.com/in/vinod-vidwans-2b57b4b/?originalSubdomain=in)
<!-- **[Paper](https://arxiv.org/abs/2502.12534), [Project Page](https://theialab.github.io/noksr/)** -->
<!-- ![noksr](assets/Teaser.png) -->

Digitizing text from historical manuscripts yields historians multiple benefits. The digitization process consists of three steps: text-line-image segmentation, text recognition from the text-line-images (and post-correction). 

This tool enables segmenting text-line-images from pages with diverse layouts. It represents text-lines as graphs, with characters as the nodes, and with edges connecting each character of a text-line to it's previous and next neighbour. In other words, we use nodes and edges as units of comparison and data collection instead of dense pixel-level metrics. This enables easier layout annotation, and improved performance compared to existing methods (as tested on a set of 15 pages with layouts of varying complexity, ranging from simple single-column and double-column layouts to layouts with pictures, footnotes, tables, interlinear writing, marginalia, text bleeding, staining, coloring, and irregular font sizes)

To recognise text content from the segmented text-line-images, we use a pre-trained text recognition model for the Devanāgarī script. The tools enables fine-tuning of the pre-trained model on specific manuscripts, which results in the model's predictions getting progressively better with more annotated data, thus also making the subsequent annotation easier - similar to active learning.

Contact kartik.niszoig at gmail for questions, comments and reporting bugs..


![Demo](demo.gif)

**Step 1**: Automatically Segment Text Line Images from Document, with the ability to manually ADD or DELETE edges for tricky edge-case page layouts. **Step 2**: Recognize the text content from the Text Line Images, make corrections, and fine tune the IMG2TEXT model

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

#### 1. Install & Configure MySQL

1. Download and install MySQL:  
   https://dev.mysql.com/downloads/installer/

2. Create a new database in MySQL:

   ```sql
   CREATE DATABASE Annotation_DB;
   ```
3. Change the databse url in backend/database/connection.py. Replace [password] with your actual MySQL root password.
   ```
   DATABASE_URL = "mysql+pymysql://root:[password]@localhost:3306/Annotation_DB"
   ```




#### Please follow the following steps to create the backend conda environment:
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




- [ ] Layout Analysis
    - [ ] better UX - better zoom in.
    - [ ] improve the graph construction algorithm
        - [ ] support footnotes written in a different orientation
        - [ ] Run line segmentation algorithm seperately for different font sizes 
        - [ ] Make the process closer to full automatic, using Graph Neural Networks
        - [ ] Process text of different font sizes seperately based on the heatmp (comments can have a different font size). (allow the user to tune the font size clustering sensitivity in the front end)
        - [ ] Enable sanskrit experts to manually decide a "reading order" of main text, comments, footnotes
        - [ ] Collect Node Features and Edge Features for GNN Training
    - [ ] If CRAFT fails in detecting the characters themselves, we should be able to add or delete nodes too (along with the edges)
    - [ ] record location of the segmented line on map as meta data.

- [ ] Recognition Model 
    - [ ] Fix bugs in fine tuning
    - [ ] after clicking fine tuning, we should go to next page, not home page
    - [ ] Better iterative "active learning" UX


- [ ] Post-Correction
    - [ ] integrate ByT5-Sanskrit with this tool and auto-correct the OCR output
    - [ ] finetune [ByT5-Sanskrit](https://huggingface.co/chronbmm/sanskrit-byt5-ocr-postcorrection) using [reinforcement learning](https://arxiv.org/abs/2501.17161)


- [ ] Other TODO
    - [ ] have two AI models attempt the same predictions. Places where they differ, or where they are uncertain need special attention.
    - [ ] Continue fixing inbuild devanagari typing mode. 
    - [ ] Integrate V2 EasyOCR English model, other scripts
    - [ ] Cuda profiling and Optimize GPU use
    - [ ] study benefits of "overdoing" finetuning: [Ref1](https://arxiv.org/pdf/2408.04809) [Ref2](https://imtiazhumayun.github.io/grokking/)

- [ ] Collaboration
    - [ ] Integrate with tools which analyze manuscripts, find links between manuscripts, connect references, concepts eg: shabda kosha

 
