import os
import torch
import imageio
import torchvision
import numpy as np
from einops import rearrange
from diffusers import AutoencoderKL, DDIMScheduler
from uniedit.diffuser_utils import UniEditPipeline
from feature_injection.spatialctrl import SpatialAttentionWithMask
from feature_injection.spatialctrl_utils import regiter_attention_editor_diffusers
from torchvision.io import read_image
from pytorch_lightning import seed_everything
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
from video_utils import prepare_frames
from lavie_models.unet import UNet3DConditionModel
from nti import NullInversion
from easydict import EasyDict
from PIL import Image

def save_videos_grid(videos: torch.Tensor, path: str, n_rows=6, save_input=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    if save_input:
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            outputs.append(x)
    else:
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration=125)

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = image.to(device)
    return image

def load_mask():
    mask_1 = []
    for i in range(config.frame_num):
        path_s = f'masks/{config.video_name}/video1_frame_{i}_mask.png'
        image = Image.open(path_s)
        image = image.convert('L')
        image_data = np.array(image)
        mask_s = np.where(image_data > 128, 1., 0.)
        mask_s = torch.from_numpy(mask_s).float().to(device)
        mask_1.append(mask_s)
    mask_1 = torch.stack(mask_1, dim=0)
    return mask_1

torch.cuda.set_device(0)  # set the GPU device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

### >>> config >>> ###
config = EasyDict()
config.video_name = 'cat-in-the-sun'
config.num_ddim_steps = 50
config.guidance_scale = 7.5
config.frame_num = 16
config.frame_stride = 1
config.uniedit_with_mask=True
config.nti = True
config.source_prompt = ''
config.target_prompt = 'A dog in the field.'
config.inverse_prompt = ''
config.n_prompt = ''
config.sd_model_path = "CompVis/stable-diffusion-v1-4"
config.lavie_model_path = "./path/to/LaVie/lavie_base.pt"
config.seed = 42

### >>> create pipeline >>> ###
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

input_path = f'videos/{config.video_name}.mp4'
input_dir = f'videos/{config.video_name}_320x512'
output_dir = f'outputs/{config.video_name}'
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_resolution = [320, 512]
crop = [0, 0, 0, 0]
prepare_frames(input_path, input_dir, image_resolution, crop)

frames = []
for frame in sorted(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, frame)
    source_image = load_image(image_path, device)
    frames.append(source_image)
frames = torch.cat(frames, dim=0)[::config.frame_stride][0:0+config.frame_num].unsqueeze(0)  # b f c h w
frames_for_save = rearrange(frames, "b f c h w -> b c f h w")
save_videos_grid(frames_for_save.cpu(), os.path.join(output_dir, "input.gif"), save_input=True)

if is_xformers_available():
    try:
        model.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(
            "Could not enable memory efficient attention. Make sure xformers is installed"
            f" correctly and a GPU is available: {e}"
        )

seed_everything(config.seed)
start_latent = torch.load("randn_start_latent_seed42.pt")

if config.uniedit_with_mask:
    mask = load_mask()
    editor = SpatialAttentionWithMask(mask_s=mask)
    regiter_attention_editor_diffusers(model, editor)
    editor.struc_ctrl_step_idx = list(range(25))
    editor.struc_ctrl_layer_idx = list(range(16))
    editor.bg_enhance_step_idx = list(range(25))
    editor.bg_enhance_layer_idx = list(range(16))
else:
    editor = SpatialAttentionWithMask(mask_s=None)
    regiter_attention_editor_diffusers(model, editor)
    editor.struc_ctrl_step_idx = list(range(20))
    editor.struc_ctrl_layer_idx = list(range(16))

assert editor.num_att_layers==32
editor.motion_editing = False
model.start_with_same_latent = True

latent_path = os.path.join(output_dir, "nti_start_latent.pt")
embeddings_path = os.path.join(output_dir, "nti_uncond_embeddings.pt")
if config.nti:
    if not os.path.exists(latent_path) or not os.path.exists(embeddings_path):
        null_inversion = NullInversion(model, frames, config.num_ddim_steps, config.guidance_scale, device, weight_dtype=torch.float32)
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(None, prompt=config.inverse_prompt, offsets=(0,0,0,0), verbose=True)
        torch.save(x_t, latent_path)
        torch.save(uncond_embeddings, embeddings_path)
    else:
        x_t = torch.load(latent_path)
        uncond_embeddings = torch.load(embeddings_path)

prompts = [config.source_prompt, config.target_prompt]
results = model(
            prompts,
            negative_prompt     = config.n_prompt,
            num_inference_steps = config.num_ddim_steps,
            guidance_scale      = config.guidance_scale,
            height              = image_resolution[0],
            width               = image_resolution[1],
            video_length        = config.frame_num,
            latents             = x_t.repeat(2, 1, 1, 1, 1),
            uncond_embeddings_pre=uncond_embeddings,
            ).videos

save_videos_grid(results[0:2], os.path.join(output_dir, "output.gif"))