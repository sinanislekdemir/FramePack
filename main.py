from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Performance optimizations for RTX 5070 (Blackwell SM 12.0, 12GB VRAM)
# Must be set before any CUDA initialization
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import json
import subprocess

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from peft import PeftModel
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked, set_attention_backend, get_cu_seqlens
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Enable performance optimizations for Blackwell GPU
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(min(8, (os.cpu_count() or 8)))
torch.set_num_interop_threads(min(8, (os.cpu_count() or 8)))

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = {
    "FramePack F1 NF4 4-bit (7GB VRAM)": {
        "type": "nf4",
        "repo": "furusu/framepack_f1_transformer_nf4",
    },
    "FramePack F1 bf16 (12GB VRAM)": {
        "type": "diffusers",
        "repo": "lllyasviel/FramePack_F1_I2V_HY_20250503",
    },
    "FramePack Legacy NF4 4-bit (7GB VRAM)": {
        "type": "nf4",
        "repo": "furusu/framepack_transformer_nf4",
    },
    "FramePack Legacy bf16 (12GB VRAM)": {
        "type": "diffusers",
        "repo": "lllyasviel/FramePackI2V_HY",
    },
}

ATTN_BACKENDS = ["auto", "flash", "cudnn", "sdpa", "sage_attention"]

# ---------------------------------------------------------------------------
# MagCache helpers (from megcache_f1.py)
# ---------------------------------------------------------------------------

def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])
    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def initialize_magcache(self, enable_magcache=True, num_steps=25, magcache_thresh=0.1, K=2, retention_ratio=0.2):
    self.enable_magcache = enable_magcache
    self.cnt = 0
    self.num_steps = num_steps
    self.magcache_thresh = magcache_thresh
    self.K = K
    self.retention_ratio = retention_ratio
    self.mag_ratios = np.array([1.0] + [1.25781, 1.08594, 1.02344, 1.00781, 1.02344, 1.00781, 1.02344, 1.05469, 0.99609, 1.03906, 1.00781, 1.01562, 1.00781, 1.02344, 1.01562, 0.98047, 1.05469, 0.98047, 0.96875, 1.03125, 0.97266, 0.9375, 0.96484, 0.78516])
    if len(self.mag_ratios) != num_steps:
        self.mag_ratios = nearest_interp(self.mag_ratios, num_steps)


def magcache_framepack_forward(
        self,
        hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance,
        latent_indices=None,
        clean_latents=None, clean_latent_indices=None,
        clean_latents_2x=None, clean_latent_2x_indices=None,
        clean_latents_4x=None, clean_latent_4x_indices=None,
        image_embeddings=None,
        attention_kwargs=None, return_dict=True
):
    if attention_kwargs is None:
        attention_kwargs = {}

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config['patch_size'], self.config['patch_size_t']
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

    hidden_states, rope_freqs = self.process_input_hidden_states(hidden_states, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices)

    temb = self.gradient_checkpointing_method(self.time_text_embed, timestep, guidance, pooled_projections)
    encoder_hidden_states = self.gradient_checkpointing_method(self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask)

    if self.image_projection is not None:
        assert image_embeddings is not None, 'You must use image embeddings!'
        extra_encoder_hidden_states = self.gradient_checkpointing_method(self.image_projection, image_embeddings)
        extra_attention_mask = torch.ones((batch_size, extra_encoder_hidden_states.shape[1]), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)
        encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

    if batch_size == 1:
        text_len = encoder_attention_mask.sum().item()
        encoder_hidden_states = encoder_hidden_states[:, :text_len]
        attention_mask = None, None, None, None
    else:
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1]
        cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q
        attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    if self.enable_magcache:
        if self.cnt == 0:
            self.accumulated_ratio = 1.0
            self.accumulated_steps = 0
            self.accumulated_err = 0

        skip_forward = False
        if self.cnt >= int(self.retention_ratio * self.num_steps) and self.cnt >= 1:
            cur_mag_ratio = self.mag_ratios[self.cnt]
            self.accumulated_ratio = self.accumulated_ratio * cur_mag_ratio
            cur_skip_err = np.abs(1 - self.accumulated_ratio)
            self.accumulated_err += cur_skip_err
            self.accumulated_steps += 1
            if self.accumulated_err <= self.magcache_thresh and self.accumulated_steps <= self.K and np.abs(1 - cur_mag_ratio) <= 0.06:
                skip_forward = True
            else:
                self.accumulated_ratio = 1.0
                self.accumulated_steps = 0
                self.accumulated_err = 0

        if skip_forward:
            hidden_states = hidden_states + self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for block_id, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(block, hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs)
            for block_id, block in enumerate(self.single_transformer_blocks):
                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(block, hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs)
            self.previous_residual = hidden_states - ori_hidden_states
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
    else:
        for block_id, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(block, hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs)
        for block_id, block in enumerate(self.single_transformer_blocks):
            hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(block, hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs)

    hidden_states = self.gradient_checkpointing_method(self.norm_out, hidden_states, temb)
    hidden_states = hidden_states[:, -original_context_length:, :]

    if self.high_quality_fp32_output_for_inference:
        hidden_states = hidden_states.to(dtype=torch.float32)
        if self.proj_out.weight.dtype != torch.float32:
            self.proj_out.to(dtype=torch.float32)

    hidden_states = self.gradient_checkpointing_method(self.proj_out, hidden_states)
    hidden_states = einops.rearrange(hidden_states, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                                     t=post_patch_num_frames, h=post_patch_height, w=post_patch_width,
                                     pt=p_t, ph=p, pw=p)

    if return_dict:
        return Transformer2DModelOutput(sample=hidden_states)
    return hidden_states,


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

print(args)

# ---------------------------------------------------------------------------
# Load shared models (same for both F1 and legacy)
# ---------------------------------------------------------------------------

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()

vae.enable_slicing()
vae.enable_tiling()

vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)

DynamicSwapInstaller.install_model(text_encoder, device=gpu)

# ---------------------------------------------------------------------------
# LLaMA LoRA adapter for text encoder abliteration
# ---------------------------------------------------------------------------

LLAMA_LORA_PRESETS = {
    "None": "",
    "reissbaker/llama-3.1-8b-abliterated-lora": "reissbaker/llama-3.1-8b-abliterated-lora",
    "Custom ...": "",
}

peft_text_encoder = None
_loaded_lora_id = None


def load_llama_lora(lora_path):
    global peft_text_encoder, _loaded_lora_id
    if not lora_path:
        if peft_text_encoder is not None:
            peft_text_encoder = None
            _loaded_lora_id = None
        return False

    if lora_path == _loaded_lora_id and peft_text_encoder is not None:
        return True

    if "/" in lora_path and not os.path.exists(os.path.join(lora_path, 'adapter_config.json')):
        download_dir = os.path.join(os.environ['HF_HOME'], 'llama_lora_adapters', lora_path.replace('/', '_'))
        if not os.path.exists(os.path.join(download_dir, 'adapter_config.json')):
            print(f'Downloading LLaMA LoRA from HuggingFace: {lora_path}')
            try:
                from huggingface_hub import snapshot_download
                download_dir = snapshot_download(repo_id=lora_path, cache_dir=os.environ['HF_HOME'],
                                                 local_dir_use_symlinks=False, local_dir=download_dir)
            except Exception as e:
                print(f'Failed to download LoRA from HF: {e}')
                peft_text_encoder = None
                _loaded_lora_id = None
                return False
        lora_path = download_dir

    if not os.path.exists(os.path.join(lora_path, 'adapter_config.json')):
        peft_text_encoder = None
        _loaded_lora_id = None
        return False

    try:
        if peft_text_encoder is not None:
            del peft_text_encoder
            torch.cuda.empty_cache()
        config_path = os.path.join(lora_path, 'adapter_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            if cfg.get('task_type') == 'CAUSAL_LM':
                cfg['task_type'] = 'FEATURE_EXTRACTION'
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                print('Patched task_type from CAUSAL_LM to FEATURE_EXTRACTION for LlamaModel compatibility')
            adapter_file = os.path.join(lora_path, 'adapter_model.safetensors')
            if os.path.exists(adapter_file):
                state = sf.load_file(adapter_file)
                if any('.model.' in k for k in state.keys()):
                    remapped = {k.replace('base_model.model.', 'base_model.'): v for k, v in state.items()}
                    sf.save_file(remapped, adapter_file)
                    print('Remapped adapter keys (stripped .model. prefix for bare LlamaModel)')
        peft_text_encoder = PeftModel.from_pretrained(text_encoder, lora_path)
        peft_text_encoder.eval()
        peft_text_encoder.requires_grad_(False)
        DynamicSwapInstaller.install_model(peft_text_encoder, device=gpu)
        _loaded_lora_id = lora_path
        print(f'Loaded LLaMA LoRA from {lora_path}')
        return True
    except Exception as e:
        print(f'Failed to load LLaMA LoRA: {e}')
        traceback.print_exc()
        peft_text_encoder = None
        _loaded_lora_id = None
        return False

# ---------------------------------------------------------------------------
# FFmpeg 2x upscale
# ---------------------------------------------------------------------------

def ffmpeg_upscale_2x(input_path, output_path, crf=16):
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('ffmpeg not found — skipping upscale')
        return False
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', 'scale=iw*2:ih*2:flags=lanczos',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        output_path,
    ]
    print(f'Upscaling video 2x: {" ".join(cmd)}')
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f'Upscaled video saved to {output_path}')
        return True
    except subprocess.CalledProcessError as e:
        print(f'ffmpeg upscale failed: {e.stderr.decode() if e.stderr else e}')
        return False

# ---------------------------------------------------------------------------
# Lazy transformer loading
# ---------------------------------------------------------------------------

_current_model_name = None
transformer = None
orig_forward = None


def load_nf4_transformer(model_info):
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for NF4 models. Install it with: pip install bitsandbytes"
        )
    model = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        model_info["repo"], torch_dtype=torch.bfloat16
    )
    for name, p in model.named_parameters():
        if not isinstance(p, bnb.nn.Params4bit) and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    return model


def load_transformer(model_name):
    global transformer, orig_forward, _current_model_name
    if _current_model_name == model_name and transformer is not None:
        return
    if transformer is not None:
        transformer.to(cpu)
        torch.cuda.empty_cache()

    model_info = MODELS[model_name]

    if model_info["type"] == "nf4":
        transformer = load_nf4_transformer(model_info)
    else:
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            model_info["repo"], torch_dtype=torch.bfloat16
        ).cpu()

    transformer.eval()
    transformer.high_quality_fp32_output_for_inference = True
    if model_info["type"] != "nf4":
        transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    orig_forward = transformer.__class__.forward
    _current_model_name = model_name
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    print(f'Loaded {model_name} — Free VRAM: {free_mem_gb:.2f} GB')

# Load default model (NF4) at startup
load_transformer("FramePack F1 NF4 4-bit (7GB VRAM)")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Mutual exclusion for cache checkboxes
# ---------------------------------------------------------------------------

def handle_magcache_change(magcache_value, teacache_value):
    if magcache_value and teacache_value:
        return gr.update(value=True), gr.update(value=False)
    return gr.update(value=magcache_value), gr.update(value=teacache_value)

def handle_teacache_change(magcache_value, teacache_value):
    if magcache_value and teacache_value:
        return gr.update(value=False), gr.update(value=True)
    return gr.update(value=magcache_value), gr.update(value=teacache_value)

def handle_lora_preset_change(preset_key):
    path = LLAMA_LORA_PRESETS.get(preset_key, "")
    return gr.update(value=path, visible=(preset_key == "Custom ..."))

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size,
           steps, cfg, gs, rs, gpu_memory_preservation, use_magcache, use_teacache,
           mp4_crf, attn_backend, model_name,
           magcache_thresh=0.1, magcache_K=3, magcache_retention_ratio=0.2,
           use_llama_lora=False, llama_lora_path="",
           use_ffmpeg_upscale=False):
    global peft_text_encoder
    is_f1 = "F1" in model_name

    if use_llama_lora and llama_lora_path:
        load_llama_lora(llama_lora_path)

    set_attention_backend(attn_backend)
    load_transformer(model_name)

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60

    job_id = generate_timestamp()
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        # --- Text encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        te = peft_text_encoder if (use_llama_lora and peft_text_encoder is not None) else text_encoder
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, te, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, te, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # --- Image processing ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # --- VAE encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)

        # --- CLIP Vision ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # --- Dtype ---
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # --- Sampling ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator("cpu").manual_seed(seed)

        if is_f1:
            # F1 forward sampling
            history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
            total_generated_latent_frames = 1
            section_iter = range(total_latent_sections)
        else:
            # Legacy inverted sampling
            num_frames = latent_window_size * 4 - 3
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0
            latent_paddings = list(reversed(range(total_latent_sections)))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            section_iter = latent_paddings

        for section_item in section_iter:
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            if is_f1:
                section_index = section_item
                is_last_section = False
                print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')
            else:
                latent_padding = section_item
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size
                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            # --- Cache setup ---
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_magcache:
                transformer.__class__.forward = magcache_framepack_forward
                transformer.__class__.initialize_magcache = initialize_magcache
                transformer.initialize_magcache(enable_magcache=True, num_steps=steps, magcache_thresh=magcache_thresh, K=magcache_K, retention_ratio=magcache_retention_ratio)
            elif use_teacache:
                transformer.__class__.forward = orig_forward
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.__class__.forward = orig_forward
                transformer.__class__.initialize_magcache = initialize_magcache
                transformer.initialize_teacache(enable_teacache=False)
                transformer.initialize_magcache(enable_magcache=False)

            # --- Callback ---
            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = (f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, '
                        f'Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). '
                        f'The video is being extended now ...')
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))

            # --- Build indices and clean latents ---
            if is_f1:
                indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                cl_start, cl_4x_idx, cl_2x_idx, cl_1x_idx, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                clean_latent_indices = torch.cat([cl_start, cl_1x_idx], dim=1)
                clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
                clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
                clean_latent_indices_4x = cl_4x_idx
                clean_latent_indices_2x = cl_2x_idx
                frames_val = latent_window_size * 4 - 3
            else:
                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                cl_pre_idx, blank_indices, latent_indices, cl_post_idx, cl_2x_idx, cl_4x_idx = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([cl_pre_idx, cl_post_idx], dim=1)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_post], dim=2)
                clean_latent_indices_4x = cl_4x_idx
                clean_latent_indices_2x = cl_2x_idx
                frames_val = num_frames

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=frames_val,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_indices_2x,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_indices_4x,
                callback=callback,
            )

            # --- Append history ---
            if is_f1:
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            else:
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            # --- VAE decode ---
            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation + 2)
                load_model_as_complete(vae, target_device=gpu)

            if is_f1:
                real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
            else:
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                if is_f1:
                    section_latent_frames = latent_window_size * 2
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                    history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
            stream.output_queue.push(('file', output_filename))

            if not is_f1 and is_last_section:
                break

        if use_ffmpeg_upscale:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(100, 'Upscaling video 2x with ffmpeg ...'))))
            upscaled_path = os.path.join(outputs_folder, f'{job_id}_2x.mp4')
            if ffmpeg_upscale_2x(output_filename, upscaled_path, crf=min(mp4_crf, 14)):
                stream.output_queue.push(('file', upscaled_path))

    except:
        traceback.print_exc()
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    stream.output_queue.push(('end', None))


# ---------------------------------------------------------------------------
# Gradio process wrapper
# ---------------------------------------------------------------------------

def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size,
            steps, cfg, gs, rs, gpu_memory_preservation, use_magcache, use_teacache,
            mp4_crf, attn_backend, model_name,
            magcache_thresh, magcache_K, magcache_retention_ratio,
            use_llama_lora, llama_lora_path,
            use_ffmpeg_upscale):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()
    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size,
              steps, cfg, gs, rs, gpu_memory_preservation, use_magcache, use_teacache,
              mp4_crf, attn_backend, model_name,
              magcache_thresh, magcache_K, magcache_retention_ratio,
              use_llama_lora, llama_lora_path,
              use_ffmpeg_upscale)

    output_filename = None
    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


# ---------------------------------------------------------------------------
# Quick prompts
# ---------------------------------------------------------------------------

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            # Top bar: model + attention
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=list(MODELS.keys()), value="FramePack F1 NF4 4-bit (7GB VRAM)",
                    label="Model", interactive=True)
                attn_selector = gr.Dropdown(
                    choices=ATTN_BACKENDS, value="sdpa",
                    label="Attention Backend", interactive=True)

            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                with gr.Row():
                    use_magcache = gr.Checkbox(label='Use MagCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=False, info='Faster speed, but often makes hands and fingers slightly worse. Only one cache can be active.')
                use_magcache.change(fn=handle_magcache_change, inputs=[use_magcache, use_teacache], outputs=[use_magcache, use_teacache])
                use_teacache.change(fn=handle_teacache_change, inputs=[use_magcache, use_teacache], outputs=[use_magcache, use_teacache])

                magcache_thresh = gr.Slider(label="MagCache Thresh", minimum=0.0, maximum=1.0, value=0.10, step=0.005, info='Decrease this value when the quality is poor.')
                magcache_K = gr.Slider(label="MagCache K", minimum=1, maximum=5, value=3, step=1, info='Decrease this value when the quality is poor.')
                magcache_retention = gr.Slider(label="MagCache Retention Ratio", minimum=0.0, maximum=1.0, value=0.2, step=0.01, info='Increase to make video more consistent with non-cached generation.')

                with gr.Row():
                    use_llama_lora = gr.Checkbox(label="Use LLaMA LoRA", value=False, info='Apply a LoRA adapter to the LLaMA text encoder (e.g. for abliteration).')
                with gr.Row():
                    llama_lora_preset = gr.Dropdown(
                        choices=list(LLAMA_LORA_PRESETS.keys()), value="None",
                        label="LLaMA LoRA Preset", interactive=True,
                        info='Select a pre-made abliteration LoRA or "Custom" to provide your own.')
                llama_lora_path = gr.Textbox(label="LLaMA LoRA Path", value="", placeholder="/path/to/lora_adapter or org/repo", info='Path to PEFT LoRA adapter, or HuggingFace repo ID. Auto-downloads from HF.', visible=False)
                llama_lora_preset.change(
                    fn=handle_lora_preset_change,
                    inputs=[llama_lora_preset],
                    outputs=[llama_lora_path])

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                seed = gr.Number(label="Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=2, maximum=128, value=4, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed.")
                use_ffmpeg_upscale = gr.Checkbox(label="Upscale final video 2x with ffmpeg", value=False, info='Upscale the final output video to 2x resolution using ffmpeg (lanczos, slow preset). Requires ffmpeg in PATH.')

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size,
           steps, cfg, gs, rs, gpu_memory_preservation, use_magcache, use_teacache,
           mp4_crf, attn_selector, model_selector,
           magcache_thresh, magcache_K, magcache_retention,
           use_llama_lora, llama_lora_path,
           use_ffmpeg_upscale]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
