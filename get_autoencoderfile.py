import torch
ckpt_path ="/data/wuhaotian/diffusionDemo/model/epoch=epoch=54-val_loss=val_loss=0.1025.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")

# 你的线性反变换： phys = norm * scale + shift
scale = 0.25055954
shift = 0.07842615
thr_phys = 0.15
thr_norm = (thr_phys - shift) / scale

# 写入缺失的 buffer（模型 state_dict 顶层）
ckpt["state_dict"]["brier_thr_norm"] = torch.tensor(thr_norm, dtype=torch.float32)
torch.save(ckpt, ckpt_path.replace(".ckpt", "_patched.ckpt"))
print("patched:", ckpt_path.replace(".ckpt", "_patched.ckpt"))