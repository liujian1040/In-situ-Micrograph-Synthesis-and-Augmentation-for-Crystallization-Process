# In-situ-Micrograph-Synthesis-and-Augmentation-for-Crystallization-Process
The code for the paper "Deep Learning-based in-situ Micrograph Synthesis and Augmentation for  Crystallization Process Image Analysis"
This is a two-step process. Firstly, use CocosNet to generate synthetic images; then train YOLOv8 and utilize trained checkpoint to analyze process images.
Training and test data: Image.zip
About CocosNet: https://github.com/microsoft/CoCosNet
About YOLO: https://github.com/ultralytics/ultralytics
About data analysis: analyze_manu.py is for information extraction of manually labeled images, analyze_segment.py is for images segmented by YOLO.
