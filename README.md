# manuscript-annotation-tool

## INSTALLATION
The application uses two AI modes: [CRAFT][https://github.com/clovaai/CRAFT-pytorch] and [EasyOCR's][https://github.com/JaidedAI] Devanagari pretrained model. CRAFT detext the locations of the characters in a page, which is used to crop out text-line-images from the page images.
The Devanagari pretrained model is then used to detect the text-content from the cropped text-line-images. 

- Download/Clone this repository

- Download devanagari.pth from [here][https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip]. 
- Put this file in the `backend/instance/models/recognition` folder.


- Download craft_mlt_pth from [here][https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth?download=true]. 
- Put this file in the `backend/instance/models/segmentation` folder.

### To run the backend

0. Open conda or miniconda terminal

1. cd into `backend`

2. import the conda environment using 
```
conda env create -f environment.yml
```
3. Activate the created environment.

4. run the application using 
```
flask run --debug
```

### To run the frontend

1. cd into `frontend`

2. Install the node packages using

```
npm install
```

3. Run the development server using 
```
npm run dev
```