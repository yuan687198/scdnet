# Self-Critical Distillation Network for Video-based Commonsense Captioning
There will be more updates in the future.

## Background
This repository is for the task of [video-based commonsense captioning](https://arxiv.org/abs/2003.05162), which aims to generate event-wise captions and meanwhile provide multiple commonsense descriptions (e.g., attribute, effect and intention) about the underlying event in the video.

## Dataset
V2C dataset consists: [data](https://pan.quark.cn/s/845f32100cf7) (Extraction code: Av5v), [vimage](https://pan.quark.cn/s/fc984904214a) (Extraction code: 6jy4), [video_y](https://pan.quark.cn/s/6bb00c3731a3) (Extraction code: v7Es), [saved_tensors](https://pan.quark.cn/s/ae21553d6f46) (Extraction code: 25Hq), [clip_features](https://pan.quark.cn/s/5ef6aeba2205) (Extraction code: Fkpi).

## Training and Evaluation
For training:
```
python train.py --cms int --batch_size 60 --epochs 2500 --num_layer 6 --dim_head 64 --dim_inner 1024 \
                --num_head 8 --dim_vis_feat 2048 --dropout 0.1 --rnn_layer 1 --checkpoint_path ./save \
                --info_json data/v2c_info.json --caption_json data/V2C_MSR-VTT_caption.json \
                --print_loss_every 20 --cuda --show_predict   
```

For evaluations:
```
bash test.sh
```
After training for 2000 epochs, the model can outperform the metrics shown in Table 2 of the paper (intention: CIDEr: 104.9; effect: CIDEr: 78.8; attribute: CIDEr: 35.1). However, to obtain the optimal performance, it is necessary to save and test checkpoints from different epochs individually. The model achieves its best performance at around 3000 epochs.




