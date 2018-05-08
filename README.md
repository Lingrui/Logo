# Logo
A deep learning architecture for brand log classification    
## Requirements
* Python   
* Tensorflow 
* Picpac
* ffmpeg
## Prepare training data
Convert video to images. 
```
e.g.
ffmpeg -i sample.mp4 -vf fps=30 img%3d.png
```
Arguments:
```
-i    Input video 
-vf   set video filters (fps: frame per second)
```

## Annotate training data
Annotate training date by category (Picpac - polygon) and export to db.    
Then merge them into one database all.db  
```Python
./merge.py
```

## Training 
### train-slim-fcn-board.py 
Trainer of images classification.  
Requirs a symbolic link from Tensorflow official [Models](https://github.com/tensorflow/models) to working directory.
```
e.g.
../train-slim-fcn-board.py --db db/all --model model --classes 7 --nocache --batch 7 \
--ckpt_epochs 5  --aaalgo --backbone tiny --noreg
```
Arguments:
```
--db    Trainning db
--model   Directory to save models
--classes   Number of classes 
--batch   Batch size 
--backbone    Network architecture
```

## Predict and generate html
### video.py
#### Generate a whole video for test 
```
e.g
./video_tmp.py --input sample.mp4  --output test.avi --fps 30 --model ./model/100 
```
#### Generate seperate videos of each brand and a html file including all the details
```
e.g.
./video_tmp.py --input sample.mp4  --output test_pkl.avi --fps 30 --model ./model/100 --record tmp.pkl 
```
Arguments:
```
--input   Input directory of video
--output    Output directory of video 
--fps   Frame per second
--model   Directory of trained models
--record    Directory of dumped pickle file 

```



