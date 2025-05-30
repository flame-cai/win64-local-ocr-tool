# manuscript-annotation-tool

## Envitonment Setup
The code is tested on Windows 11 (x64), PyTorch 2.4.1, with CUDA 12.8 Driver, and cuda-12.1.1 Runtime, with Python 3.11. Please follow the following steps to install:
```

```
# Download/Clone this repository
git clone https://github.com/flame-cai/win64-local-ocr-tool.git
cd win64-local-ocr-tool
```

The application uses two AI modes: [CRAFT](https://github.com/clovaai/CRAFT-pytorch) and [EasyOCR's](https://github.com/JaidedAI) Devanagari pretrained model. CRAFT detext the locations of the characters in a page, which is used to crop out text-line-images from the page images.
The Devanagari pretrained model is then used to detect the text-content from the cropped text-line-images. 

- Download devanagari.pth from [here](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip). 
- Put this file in the `backend/instance/models/recognition` folder.

- Download craft_mlt_pth from [here](https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth?download=true). 
- Put this file in the `backend/instance/models/segmentation` folder.

### To run the backend
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

### To run the frontend
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
- finetune [ByT5-Sanskrit](https://huggingface.co/chronbmm/sanskrit-byt5-ocr-postcorrection) using [reinforcement learning](https://arxiv.org/abs/2501.17161)
- integrate ByT5-Sanskrit with this tool and auto-correct the OCR output
 
