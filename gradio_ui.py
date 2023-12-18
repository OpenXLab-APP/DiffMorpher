import os
import torch
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from datetime import datetime
from morph_attn import DiffMorpherPipeline
from lora_utils import train_lora

LENGTH=512

def train_lora_interface(
    image_0,
    image_1,
    prompt_0,
    prompt_1,
    model_path,
    save_lora_dir,
    output_path,
    lora_steps,
    lora_rank,
    lora_lr
):
    os.makedirs(save_lora_dir, exist_ok=True)
    train_lora(image_0, prompt_0, save_lora_dir, model_path,
               lora_steps=lora_steps, lora_lr=lora_lr, lora_rank=lora_rank, weight_name=f"{output_path.split('/')[-1]}_lora_0.ckpt", progress=gr.Progress())
    train_lora(image_1, prompt_1, save_lora_dir, model_path,
            lora_steps=lora_steps, lora_lr=lora_lr, lora_rank=lora_rank, weight_name=f"{output_path.split('/')[-1]}_lora_1.ckpt", progress=gr.Progress())
    return "Train LoRA Done!"

def run_diffmorpher(
    image_0,
    image_1,
    prompt_0,
    prompt_1,
    model_path,
    lora_mode,
    lamb,
    use_adain,
    use_reschedule,
    num_frames,
    fps,
    save_lora_dir,
    load_lora_path_0,
    load_lora_path_1,
    output_path
):
    run_id = datetime.now().strftime("%H%M") + "_" +  datetime.now().strftime("%Y%m%d")
    os.makedirs(output_path, exist_ok=True)
    morpher_pipeline = DiffMorpherPipeline.from_pretrained(model_path, torch_dtype=torch.float32).to("cuda")
    if lora_mode == "Fix LoRA 0":
        fix_lora = 0
    elif lora_mode == "Fix LoRA 1":
        fix_lora = 1
    else:
        fix_lora = None
    if not load_lora_path_0:
        load_lora_path_0 = f"{save_lora_dir}/{output_path.split('/')[-1]}_lora_0.ckpt"
    if not load_lora_path_1:
        load_lora_path_1 = f"{save_lora_dir}/{output_path.split('/')[-1]}_lora_1.ckpt"
    images = morpher_pipeline(
        img_0=image_0,
        img_1=image_1,
        prompt_0=prompt_0,
        prompt_1=prompt_1,
        load_lora_path_0=load_lora_path_0,
        load_lora_path_1=load_lora_path_1,
        lamb=lamb,
        use_adain=use_adain,
        use_reschedule=use_reschedule,
        num_frames=num_frames,
        fix_lora=fix_lora
    )
    video_path = f"{output_path}/{run_id}.mp4"
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (LENGTH, LENGTH))
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()
    cv2.destroyAllWindows()
    return output_video.update(value=video_path)

with gr.Blocks() as demo:
    
    with gr.Row():
        gr.Markdown("""
        # Official Implementation of [DiffMorpher](https://kevin-thu.github.io/DiffMorpher_page/)
        """)

    original_image_0, original_image_1 = gr.State(None), gr.State(None)
    # key_points_0, key_points_1 = gr.State([]), gr.State([])
    # to_change_points = gr.State([])
    
    with gr.Row():
        with gr.Column():
            input_img_0 = gr.Image(type="numpy", label="Input image 0", show_label=True, height=LENGTH, width=LENGTH, interactive=True)
            prompt_0 = gr.Textbox(label="Prompt for image 0", interactive=True)
            train_lora_button = gr.Button("Train LoRA")
            # show_correspond_button = gr.Button("Show correspondence points")
        with gr.Column():
            input_img_1 = gr.Image(type="numpy", label="Input image 1 ", show_label=True, height=LENGTH, width=LENGTH, interactive=True)
            prompt_1 = gr.Textbox(label="Prompt for image 1", interactive=True)
            clear_button = gr.Button("Clear All")
        with gr.Column():
            output_video = gr.Video(format="mp4", label="Output video", show_label=True, height=LENGTH, width=LENGTH, interactive=False)
            lora_progress_bar = gr.Textbox(label="Display LoRA training progress", interactive=False)
            run_button = gr.Button("Run!")
        # with gr.Column():
        #     output_video = gr.Video(label="Output video", show_label=True, height=LENGTH, width=LENGTH)
        
    with gr.Accordion(label="Algorithm Parameters"):
        with gr.Tab("Basic Settings"):
            with gr.Row():
                # local_models_dir = 'local_pretrained_models'
                # local_models_choice = \
                #     [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
                model_path = gr.Text(value="stabilityai/stable-diffusion-2-1-base",
                    label="Diffusion Model Path", interactive=True
                )
                lamb = gr.Slider(value=0.6, minimum=0, maximum=1, step=0.1, label="Lambda for attention replacement", interactive=True)
                lora_mode = gr.Dropdown(value="LoRA Interp",
                    label="LoRA Interp. or Fix LoRA",
                    choices=["LoRA Interp", "Fix LoRA 0", "Fix LoRA 1"],
                    interactive=True
                )
                use_adain = gr.Checkbox(value=True, label="Use AdaIN", interactive=True)
                use_reschedule = gr.Checkbox(value=True, label="Use Reschedule", interactive=True)
            with gr.Row():
                num_frames = gr.Number(value=25, minimum=0, label="Number of Frames", precision=0, interactive=True)
                fps = gr.Number(value=10, minimum=0, label="FPS (Frame rate)", precision=0, interactive=True)
                output_path = gr.Text(value="./results", label="Output Path", interactive=True)
                
        with gr.Tab("LoRA Settings"):
            with gr.Row():
                lora_steps = gr.Number(value=200, label="LoRA training steps", precision=0, interactive=True)
                lora_lr = gr.Number(value=0.0002, label="LoRA learning rate", interactive=True)
                lora_rank = gr.Number(value=16, label="LoRA rank", precision=0, interactive=True)
                save_lora_dir = gr.Text(value="./lora", label="LoRA model save path", interactive=True)
                load_lora_path_0 = gr.Text(value="", label="LoRA model load path for image 0", interactive=True)
                load_lora_path_1 = gr.Text(value="", label="LoRA model load path for image 1", interactive=True)
    
    def store_img(img):
        image = Image.fromarray(img).convert("RGB").resize((512,512), Image.BILINEAR)
        # resize the input to 512x512
        # image = image.resize((512,512), Image.BILINEAR)
        # image = np.array(image)
        # when new image is uploaded, `selected_points` should be empty
        return image
    input_img_0.upload(
        store_img,
        [input_img_0],
        [original_image_0]
    )
    input_img_1.upload(
        store_img,
        [input_img_1],
        [original_image_1]
    )
    
    def clear(LENGTH):
        return gr.Image.update(value=None, width=LENGTH, height=LENGTH), \
            gr.Image.update(value=None, width=LENGTH, height=LENGTH), \
            None, None, None, None
    clear_button.click(
        clear,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [input_img_0, input_img_1, original_image_0, original_image_1, prompt_0, prompt_1]
    )
        
    train_lora_button.click(
        train_lora_interface,
        [
         original_image_0,
         original_image_1,
         prompt_0,
         prompt_1,
         model_path,
         save_lora_dir,
         output_path,
         lora_steps,
         lora_rank,
         lora_lr
        ],
        [lora_progress_bar]
    )
    
    run_button.click(
        run_diffmorpher,
        [
         original_image_0,
         original_image_1,
         prompt_0,
         prompt_1,
         model_path,
         lora_mode,
         lamb,
         use_adain,
         use_reschedule,
         num_frames,
         fps,
         save_lora_dir,
         load_lora_path_0,
         load_lora_path_1,
         output_path
        ],
        [output_video]
    )
        
demo.queue().launch(debug=True)
