
Trained a image captioning LSTM model with youcook2 dataset.

# requiremnt
  1. speechRecognition
  2. pydub
  3. and other package in original readme
  4. ffmpeg 
  5. Potential issue with windows .. i used linux
  - Python 3
- PyTorch 1.3+ (along with torchvision)
- cider (already been added as a submodule)
- yacs
- lmdbdict
-tensorboard
- place resnet101 model inside data/image_weights


This repo is forked , see reference.

# root Structure
```

root
  -project folder (unzip the data.zip and rename it to project) 
  -self-critical
    - this repo without data.zip 
    - replace some of the files with the file inside (fix.zip) 
        specifially , tools/train.py , eval.py etc ...
  -raw_caption
  -raw_video
```
# step 1 : Video processing 
  1. assume all videos are in the raw-videos folder (in the root folder where project folder will be)
  2. run mp3wave.py to extract wav
  3. run frame.py to extract jpg 
  4. run gooogle.py to extract caption for each image
  5. run dataset.py to conver to coco 
  
# training
  
### Prepare data.
cd into self-critical folder do the following 
 python scripts/prepro_labels.py --input_json data/data.json --output_json data/datatalk.json --output_h5 data/datatalk
 python scripts/prepro_ngrams.py --input_json data/data.json --dict_json data/datatalk.json --output_pkl data/data-train --split train
 python scripts/prepro_reference_json.py --input_json data/data.json --output_json data/data_captions4eval.json
 python scripts/prepro_feats.py --input_json data/data.json --output_dir data/datatalk --images_root ../raw_images

### Start training

```
python tools/train.py --id fc --caption_model newfc --input_json data/datatalk.json --input_fc_dir data/datatalk_fc --input_att_dir data/datatalk_att --input_label_h5 data/datatalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 5000 --val_images_use 10 --max_epochs 5

```

output model.pth will be saved in opt-fc folder

## Generate image captions

### Evaluate on raw images

**Note**: this doesn't work for models trained with bottomup feature.
Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

change the model.pth and  infos.pkl path 

```bash
$ python tools/eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```

The eval script will create an `vis.json` file inside the `vis` folder

## Reference

```
@article{luo2018discriminability,
  title={Discriminability objective for training descriptive captions},
  author={Luo, Ruotian and Price, Brian and Cohen, Scott and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1803.04376},
  year={2018}
}
```
Of course, please cite the original paper of models you are using (You can find references in the model files).

## Acknowledgements

Thanks the original [neuraltalk2](https://github.com/karpathy/neuraltalk2) and awesome PyTorch team.
