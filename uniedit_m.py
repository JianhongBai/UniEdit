import os
import torch
import imageio
import torchvision
import numpy as np
from einops import rearrange
from diffusers import AutoencoderKL, DDIMScheduler
from uniedit.diffuser_utils import UniEditPipeline
from feature_injection.spatialctrl_utils import regiter_attention_editor_diffusers
from feature_injection.tempctrl_utils import regiter_temp_attention_editor_diffusers
from feature_injection.spatialctrl import SpatialAttentionWithMask
from feature_injection.tempctrl import TempAttentionWithMask
from torchvision.io import read_image
from pytorch_lightning import seed_everything
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
from lavie_models.unet import UNet3DConditionModel
from PIL import Image
from easydict import EasyDict

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, save_input=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration=125)

torch.cuda.set_device(0)  # set the GPU device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

### >>> config >>> ###
config = EasyDict()
config.video_name = 'motion'
config.num_ddim_steps = 50
config.guidance_scale = 7.5
config.frame_num = 16
config.frame_stride = 1
config.source_prompt = 'A cute raccoon playing guitar in the park at sunrise, oil painting style.'
config.target_prompt = 'A cute raccoon eating an apple in the park at sunrise, oil painting style.'
config.n_prompt = ''
config.sd_model_path = "stable-diffusion-v1-4"
config.lavie_model_path = "./LaVie/lavie_base.pt"
config.seed = 42

### >>> create validation pipeline >>> ###
tokenizer    = CLIPTokenizer.from_pretrained(config.sd_model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(config.sd_model_path, subfolder="text_encoder")
vae          = AutoencoderKL.from_pretrained(config.sd_model_path, subfolder="vae")
scheduler = DDIMScheduler.from_pretrained(config.sd_model_path,
                                        subfolder="scheduler",
                                        beta_start=0.0001, 
                                        beta_end=0.02, 
                                        beta_schedule="linear")            
unet = UNet3DConditionModel.from_pretrained_2d(config.sd_model_path, subfolder="unet")

checkpoint = torch.load(config.lavie_model_path, map_location=lambda storage, loc: storage)
if "ema" in checkpoint:
    print('Ema existing!')
    checkpoint = checkpoint["ema"]
unet.load_state_dict(checkpoint)
model = UniEditPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler).to("cuda")
model = model.to("cuda")
model.unet.eval()
model.vae.eval()
model.text_encoder.eval()
if is_xformers_available():
    try:
        model.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(
            "Could not enable memory efficient attention. Make sure xformers is installed"
            f" correctly and a GPU is available: {e}"
        )

output_dir = f'outputs/{config.video_name}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

seed_everything(config.seed)
image_resolution = [512, 320]
start_latent = torch.load("randn_start_latent_seed42.pt")

def load_mask():
    def load_mask_type(mask_type):
        masks = []
        for i in range(config.frame_num):
            path = f'masks/{config.video_name}/{config.target_prompt}_{mask_type}_frame_{i}.png'
            image = Image.open(path).convert('L')
            mask = np.where(np.array(image) > 128, 1., 0.)
            mask = torch.from_numpy(mask).float().to(device)
            masks.append(mask)
        return torch.stack(masks, dim=0)
    mask_1 = load_mask_type('content_mask')
    mask_2 = load_mask_type('source_fg_mask')
    mask_3 = load_mask_type('target_fg_mask')
    return mask_1, mask_2, mask_3

mask_1, mask_2, mask_3 = load_mask()
mask_t = (mask_3.bool() | mask_1.bool()).float()

editor = SpatialAttentionWithMask(mask_s=mask_1, mask_t=mask_t)
temp_editor = TempAttentionWithMask()

regiter_attention_editor_diffusers(model, editor)
regiter_temp_attention_editor_diffusers(model, temp_editor)
assert editor.num_att_layers==32
assert temp_editor.num_att_layers==16

editor.struc_ctrl_step_idx = list(range(15))
editor.struc_ctrl_layer_idx = list(range(16))
editor.content_pre_step_idx = list(range(4, 50))
editor.content_pre_layer_idx = list(range(10, 16))
editor.mask_s_fg = mask_2
editor.motion_editing = True

prompts = [config.source_prompt, config.target_prompt, config.target_prompt]
results = model(
                prompts,
                negative_prompt     = config.n_prompt,
                num_inference_steps = config.num_ddim_steps,
                guidance_scale      = config.guidance_scale,
                height              = image_resolution[1],
                width               = image_resolution[0],
                video_length        = config.frame_num,
                latents             = start_latent.repeat(3, 1, 1, 1, 1),
                ).videos
save_videos_grid(results[0:2], os.path.join(output_dir, "output.gif"))