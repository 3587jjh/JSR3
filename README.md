# PCSR (Pixel-wise Cascading SR)
# Config Modification
- cd configs/
- x = ablation number = 1,2,3,4 ...
- cp carn-liif-pcsr-x4-phase2-A.yaml carn-liif-pcsr-x4-phase2-B-x.yaml<br/><br/>
- Modify inside the carn-liif-pcsr-x4-phase2-B-x.yaml:
  - loss_rgb_w: p, loss_avg_w: q, loss_ce_w: r
  - resume_path: save/carn-liif-pcsr-x4-phase2-B-x/iter_last.pth

# Train
- python train_pcsr.py --config configs/carn-liif-pcsr-x4-phase2-B-x.yaml --gpu 0 [--resume]

# Test
- python test_pcsr.py --config configs/carn-liif-pcsr-x4-phase2-B-x.yaml --batch_size 36 --diff_threshold 0.7
