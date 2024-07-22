# Debugging using IP-adapter.ipynb

import torch

def encode_image(image, image_encoder, feature_extractor, num_images_per_prompt=1):
    dtype = next(image_encoder.parameters()).dtype
    device = next(image_encoder.parameters()).device

    if not isinstance(image, torch.Tensor):
        image = feature_extractor(image, return_tensors="pt").pixel_values # [1, 3, 224, 224]
    
    image = image.to(device=device, dtype=dtype)
    image_embeds = image_encoder(image).image_embeds # (B, D)
    image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0) # (B*num_images_per_prompt, D)

    return image_embeds


def _prepare_ip_adapter_image_embeds(
    self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
):
    ip_adapter_image = None
    device = None
    num_images_per_prompt = 1 # !!!!!!!!!! one2one mapping for img and text embeds
    
    repeat_dims = [1]
    image_embeds = []
    for single_image_embeds in ip_adapter_image_embeds:
        if do_classifier_free_guidance:
            single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
            single_image_embeds = single_image_embeds.repeat(
                num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
            )
            single_negative_image_embeds = single_negative_image_embeds.repeat(
                num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
            )
            single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
        else:
            single_image_embeds = single_image_embeds.repeat(
                num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
            )
        image_embeds.append(single_image_embeds)

    return image_embeds

class Generator4Embeds:

    def __init__(self, path="stabilityai/sdxl-turbo", num_inference_steps=1, device=torch.device('cuda')):
        # path: "stabilityai/sdxl-turbo" or "stabilityai/stable-diffusion-xl-base-1.0" or your local path

        import os
        os.environ['http_proxy'] = 'http://10.16.35.10:13390' 
        os.environ['https_proxy'] = 'http://10.16.35.10:13390' 

        self.num_inference_steps = num_inference_steps
        self.dtype = torch.float16
        self.device = device
        
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, variant="fp16").to(device)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl_vit-h.bin", torch_dtype=torch.float16)
        self.pipe = pipe
        self.pipe.prepare_ip_adapter_image_embeds = _prepare_ip_adapter_image_embeds.__get__(self.pipe)

    def generate(
            self, 
            image_embeds,
            text_prompt='', 
            num_inference_steps=None, 
            negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
            guidance_scale=None,
            **kwargs,
        ):
        # sdxl-turbo: Make sure to set guidance_scale to 0.0 to disable, as the model was trained without it.
        num_inference_steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        if guidance_scale is None:
            guidance_scale = 0.0 if 'turbo' in self.pipe._name_or_path else 5
        image_embeds_prompt = self.make_image_embeds_prompt(image_embeds, guidance_scale)
        num_images_per_prompt = image_embeds.shape[0]

        # generate image with image prompt - ip_adapter_embeds
        image = self.pipe(
            prompt=text_prompt, 
            ip_adapter_image_embeds=image_embeds_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        ).images

        return image
    
    def make_image_embeds_prompt(self, image_embeds, guidance_scale):
        # image_embeds: (B, D) divide to list of list[1*(B * 1/2, 1, D)]
        image_embeds = image_embeds.unsqueeze(1) # (B, 1, D)
        if guidance_scale > 1: #  or 'turbo' in self.pipe._name_or_path
            negative_image_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_image_embeds, image_embeds]) # (2 * B, 1, D)
        return [image_embeds]
    

if __name__ == '__main__':
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    import torch

    feature_extractor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", 
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe_image = Generator4Embeds(path='stabilityai/sdxl-turbo', num_inference_steps=4)
    
    from diffusers.utils import load_image
    image_prompt = load_image("/mnt/dataset0/weichen/projects/visobj/proposals/mise/data/things-images/THINGSplus/images/images_resized/cat.jpg")
    from IPython.display import Image, display
    display(image_prompt)
    image_embeds = encode_image(image_prompt, image_encoder, feature_extractor)

    pipe_image.pipe.set_ip_adapter_scale(0.5)
    generator = torch.Generator().manual_seed(0)
    images = pipe_image.generate(
        image_embeds=image_embeds,
        text_prompt="a cat sitting in a car",
        negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
    )
    display(images)