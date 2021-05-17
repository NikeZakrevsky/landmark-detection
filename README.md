# Landmark detection


## Preparation

Download [model.pth](https://drive.google.com/file/d/1R3ni1BmT0EDvVRgv3l0-np_wMgif3Hc0/view?usp=sharing) from Google Drive, and extract the downloaded file into ~/checkpoint/.

## Evaluation
```
python demo.py --image Image.png --face 432 575 819 971 --save_path result.png
```

## Testing
```
cd tests; pytest
```
