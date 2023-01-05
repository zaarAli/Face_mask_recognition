This is a file that instructs you to run the demo.

Installation

1. $ conda create -n mask_de
2. $ conda activate mask_de
3. $ cat requirements.txt | xargs -n 1 conda install
4. $ conda install pip
5. $ cat requirements.txt | xargs -n 1 pip install  

Run demo:

1. $ conda activate mask_de
2. $ python detect_cam.py -m mask-detector.model    


Run heatmap:

1. $ conda activate mask_de
2. $ python heatmap_cam.py -m mask-detector.model    

