import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs, output_attentions=False):
        # return image_forward_outs.hidden_states[:-1], image_forward_outs.attentions
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if output_attentions:
            image_attentions = image_forward_outs.attentions[-1]
            image_attentions = image_attentions.mean(dim=-2)
            return image_features, image_attentions
        return image_features

    @torch.no_grad()
    def forward(self, images, output_attentions=False):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), 
                                                   output_hidden_states=True, output_attentions=output_attentions)
            image_features = self.feature_select(image_forward_outs, output_attentions=output_attentions)
            if not isinstance(image_features, tuple):
                image_features = image_features.to(images.dtype)
            else:
                image_features = (image_features[0].to(images.dtype), image_features[1].to(images.dtype))

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2