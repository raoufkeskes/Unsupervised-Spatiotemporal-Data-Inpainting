# Unsupervised-Spatiotemporal-Data-Inpainting
Inpainting occlusions in images has been always a fascinating hard task for humans , the latter is harder when we extend it to videos and add a temporal axis . Researchers have been working on it for several decades achieving quite good results in a classical way but the texture was always not very smooth and somtimes blurry . However , recently with the success of "Deep Learning" , DL models have become ubiquitous in all tasks including **Inpainting** and especially GANs introduced in 2014 by 'Ian GOODFELLOW' inspired from Game theory which turn out to perform very well in this kind of problems .

Therefore , this repository is about the implementation of the paper : https://openreview.net/forum?id=rylqmxBKvH tackling this problem in **an unsupervised context** titled "Unsupervised Spatiotemporal Data Inpainting" where we aim to implement it from scratch and reproduce experiments and results .

The main idea of the paper is **to inpaint occlusions on geophysical and natural videos sequencies in a fully unsupervised context** , where they introduce GANs paradigm in their architectures to solve this problem from a probabilistic viewpoint .



![alt text][logo]

[logo]: https://i.ibb.co/bmK0LWc/Unsupervised-Inpainting-GAN-based.png 

## Requirements
``` pip install -r requirements.txt ```
## Usage 
starting 
1) ```git clone https://github.com/raoufkeskes/Unsupervised-Spatiotemporal-Data-Inpainting.git```
2) ```cd Unsupervised-Spatiotemporal-Data-Inpainting```

### Data 
the data will be stored on a folder ```datasets``` external to the project repo folder to avoid missleading pushes 
```
 parent directory
    ├── Unsupervised-Spatiotemporal-Data-Inpainting (our repo) 
    └── datasets
```
* **FaceForensics** : This dataset contains 1000 videos of non-occluded facemovements on a static background ,
two options are available to get the data   
    * download extracted faces ready to use : 
    ```  bash data/scripts/download_preprocess_FaceForensics.sh ```
    *  download original videos (1000 youtube videos) and apply face recognition  from scratch (it could take a while many hours)
    ```  bash data/scripts/download_preprocess_FaceForensics.sh from_scratch```

* **KTH** : A human action dataset containing 2391 video clips of 6 human actions, to get it just execute the corresponding script 
```  bash data/scripts/download_preprocess_FaceForensics.sh ```

### Run training
Depending on your ressources you have to adjust the number of frames (35 frames is the best number according to authors) 
``` python train.py --root ../datasets/KTH/ --num_frames 10 ```

### Metrics
#### FID: Frechet Inception Distance
calculate the the Frechet Inception Distance (FID) given the batch size and the two paths to the dataset. Or, path_real can be the pre calculated mean and sigma of the real dataset.

```
python metrics/fid.py -pr path1 -pg path2
```
OR 
```
import metrics as m

fid_score = m.fid.getFID(path_real, path_gen, batch_size)
```

#### FVD: Frechet Video Distance
calculate the the Frechet Video Distance (FVD) given the two paths to the dataset.
OTHERS TO DO
```
python metrics/fvd.py -pr path1 -pg path2
```
OR 
```
import metrics as m

fvd_score = m.fvd.getFVD(path_real, path_gen)
```


<!---                                 
## Notes
TO DO
## Experiments and results 
TO DO paper
TO DO ours ( if we have time ) 
## Proposed improvements 
for the paper : 
for the implementation : 
-->

## Licence 
January , 2020
[MIT license](http://opensource.org/licenses/MIT).

## References
[1] Yuan Yin, Arthur Pajot, Emmanuel de Bézenac, Patrick Gallinari , "Unsupervised Spatiotemporal Data Inpainting" , URL : https://openreview.net/forum?id=rylqmxBKvH







