# This is a Demo to test our VAPNet model.

## Environment

We recommend using anaconda to create the environment.

```
conda env create -f environment.yaml
```

## Dataset

Please download related datasets: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

For Video Caption model, please refer to [PDVC](https://github.com/ttengwang/PDVC). It is worth noting that we retrain the visual backbone and caption model with no overlap classes between UCF101 and HMDB51.

For convience of reproduction, we provide pre-extracted features of video from R(2+1)D, category name/related descriptions/captions from SBERT.

Data folder structure:

```bash
data/
├── HMDB51
│   ├── hmdb_cap_clean.npy
│   ├── hmdb_cls.npy
│   ├── hmdb_des.npy
│   ├── hmdb_video_feature_1.npy
│   ├── hmdb_video_feature_25.npy
│   └── hmdb_video_label.npy
└── UCF101
    ├── ucf_cap_clean.npy
    ├── ucf_cls.npy
    ├── ucf_des.npy
    ├── ucf_video_feature_1.npy
    ├── ucf_video_feature_25.npy
    └── ucf_video_label.npy
```

## Checkpoints

We provide our trained VAPNet in the [checkpoints](checkpoints) folder.

## Test

You can execute the following comman to reproduce the result:

For UCF101 with video_clip_num=1:

```python3
python test.py --dataset UCF101 --clip_num 1 --num_heads 16 --checkpoint checkpoints/VAPNet662_checkpoint_UCF_16.pth
```

For UCF101 with video_clip_num=25:

```python3
python test.py --dataset UCF101 --clip_num 25 --num_heads 16 --checkpoint checkpoints/VAPNet662_checkpoint_UCF_16.pth
```

For HMDB51 with video_clip_num=1:

```python3
python test.py --dataset HMDB51 --clip_num 1 --num_heads 16 --checkpoint checkpoints/VAPNet662_checkpoint_HMDB_16.pth
```

For HMDB51 with video_clip_num=25:

```python3
python test.py --dataset HMDB51 --clip_num 25 --num_heads 16 --checkpoint checkpoints/VAPNet662_checkpoint_HMDB_16.pth
```

The results are logged in [result.log](result.log), which are shown in Table 3 (see VAPNet) of the manuscript.

