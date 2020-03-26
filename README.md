# Abstract
The project was developed as a base technology for autonomous train traffic, with only the areas corresponding to the tracks from the cameras installed at the front of the trains. This Repository was created in 2019 as part of Soongsil University's industry-academic cooperation project. The architecture and code will be used to create the thesis, and you can use them as you wish after the license is specified.
# Defendency
```
pip install -r requirements.txt
```
# Execute
```
python main.py --input video/test1.mp4
```
# Architecture
<img src="/introduce/architecture.jpg">
# Process
<img src="/introduce/process.png">
From the top, the original images, black-and-white images, histogram smoothing images, blurry images, motionblur images, convolution images, sliding windows and prediction lines are shown in order.
# Execution Example
<img src="/introduce/1.JPG"><img src="/introduce/2.JPG">
<img src="/introduce/3.JPG"><img src="/introduce/4.JPG">
<img src="/introduce/5.JPG">
