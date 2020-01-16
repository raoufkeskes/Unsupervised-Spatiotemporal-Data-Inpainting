# Unsupervised-Spatiotemporal-Data-Inpainting
Inpainting occlusions in images has been always a fascinating hard task for humans , the latter is harder when we extend it to videos and add a temporal axis . Researchers have been working on it for several decades achieving quite good results in a classical way but the texture was always not very smooth and somtimes blurry . However , recently with the success of "Deep Learning" , DL models have become ubiquitous in all tasks including **Inpainting** and especially GANs introduced in 2014 by 'Ian GOODFELLOW' inspired from Game theory which turn out to perform very well in this kind of problems .

Therefore , this repository is about the implementation of an ICLR 2020 accepted paper : https://openreview.net/forum?id=rylqmxBKvH tackling this problem in **an unsupervised context** titled "Unsupervised Spatiotemporal Data Inpainting" where we aim to implement it from scratch and reproduce experiments and results .

The main idea of the paper is **to inpaint occlusions on geophysical and natural videos sequencies in a fully unsupervised context** , where they introduce GANs paradigm in their architectures to solve this problem from a probabilistic viewpoint .

## Requirements
``` pip install -r requirements.txt ```
## Usage 

### Data 
#### Download the data

```bash Data/ALL_IN_ONE_data_downloader.sh``` to download everything at once ( **Note** : BAIR data size is +30GB , thus we recommend to not download it for the first time while testing the app and getting it touch with it ... )

```bash Data/ALL_IN_ONE_data_downloader.sh FaceForensics KTH SST```
#### Apply preprocessing + occlusions
``` python Data/preprocess.py [Args]* ```
Arguments are : 
* ```--datasets```   : datasets to process    , **default = "FaceForensics,KTH"**
* ```--occlusions``` : occlusions to perform  , **default = ""moving_bar,raindrops,remove_pixels""**
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

typical tree for datasets is : 
```/Data/datasets/$dataset_name/```

                            |-- raw-data
                            
                            |-- occluded-data
                            
                            |-- resized-data
                                    
After doing the 2 steps cited above try to explore this folder for a better understanding of how the data is organized .

## Notes

TO DO

## Experiments and results 

TO DO paper

TO DO ours ( if we have time ) 

## Proposed improvements 
for the paper : 

for the implementation : 

## Licence 
December , 2019
[MIT license](http://opensource.org/licenses/MIT).

## References
[1] Yuan Yin, Arthur Pajot, Emmanuel de Bézenac, Patrick Gallinari , "Unsupervised Spatiotemporal Data Inpainting" , International Conference on Learning Representations (ICLR 2020) , URL : https://openreview.net/forum?id=rylqmxBKvH







