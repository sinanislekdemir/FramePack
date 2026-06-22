# WAN 2.1 Iterative Video Generation (I2V 480P)

This demo adapts the FramePack-F1 algorithm for WAN 2.1 Image-to-Video model.

## Features

- **Iterative Generation**: Creates longer videos by generating 1-second segments
- **Last Frame Chaining**: Uses the last frame of each segment as input for the next
- **Smooth Blending**: Soft transitions between segments using frame blending
- **Auto Resolution**: Automatically scales and crops images to optimal buckets
- **Memory Optimized**: Configured for 12GB VRAM with CPU offloading

## Model

- **Model**: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- **Size**: ~14GB (FP16)
- **Resolution**: 480P (288x512 to 672x384)
- **FPS**: 16fps
- **Segment Length**: 16 frames (1 second)

## Requirements

- 12GB+ VRAM (tested with CPU offloading)
- Python 3.10+
- diffusers 0.33.1+
- See `requirements.txt`

## Usage

```bash
python demo_gradio_wan21.py --inbrowser
```

### Command Line Options

- `--share`: Create a public Gradio share link
- `--server SERVER`: Server address (default: 0.0.0.0)
- `--port PORT`: Port number
- `--inbrowser`: Automatically open browser

## Algorithm

1. Load input image → Scale & crop to optimal bucket size
2. For each second to generate:
   - Generate 16-frame video from current frame + prompt
   - Extract last frame
   - Use as input for next iteration
   - Blend overlapping frames for smoothness
3. Save accumulated video segments as MP4

## Memory Tips

If you get OOM errors:
- Reduce "Total Video Length" (try 2-3 seconds first)
- Increase "GPU Memory Preservation" slider
- Close other GPU applications
- The first generation is always slower (model loading)

## Comparison with F1

| Feature | FramePack-F1 | WAN 2.1 I2V |
|---------|-------------|-------------|
| Base Model | HunyuanVideo | WAN 2.1 |
| Resolution | 640 base | 480 base |
| Complexity | High (latent windows) | Low (simple iteration) |
| Memory | 60GB+ | 12GB+ (with offload) |
| Speed | Faster per frame | Slower per frame |

## Credits

- Algorithm inspired by FramePack-F1
- WAN 2.1 model by Alibaba PAI
- Implementation using diffusers library
