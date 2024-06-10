"""
Copyright (c) 2024. Toshihiro Ota
Licensed under the Apache License, Version 2.0 (the "License");

This impl is based on `mlp_mixer` provided in timm by Ross Wightman.

Hacked together by / Copyright 2021 Ross Wightman
"""
import math
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, lecun_normal_
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

from .layers import vMlp, sMlp, aMlp, vMixerBlock, pMixerBlock

__all__ = ['vMixerBlock', 'pMixerBlock', 'MlpMixer']  # model_registry will add each entrypoint fn to this


class MlpMixer(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=vMixerBlock,
            mlp_layer=vMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            global_pool='avg',
            hidden_ratio=1,
            n_iter=1,
            lcoeff=0.,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.grad_checkpointing = False

        self.lcoeff = lcoeff

        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
        )
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        # FIXME drop_block (maybe no need, should be removed)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim,
                self.stem.num_patches,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=drop_path_rate,
                hidden_ratio=hidden_ratio,
                n_iter=n_iter,
            )
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict


def _create_mixer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        MlpMixer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'mixer_s32_224.untrained': _cfg(),
    'mixer_s16_224.untrained': _cfg(),
    'mixer_b32_224.untrained': _cfg(),
    'mixer_b16_224.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    ),
    'mixer_b16_224.goog_in21k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224_in21k-617b3de2.pth',
        num_classes=21843
    ),
    'mixer_l32_224.untrained': _cfg(),
    'mixer_l16_224.goog_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    ),
    'mixer_l16_224.goog_in21k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth',
        num_classes=21843
    ),

})


# vanilla Mixer models
@register_model
def vanillamixer_s32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=1,
                      **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def vanillamixer_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def vanillamixer_s16_224_ni2(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=2,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def vanillamixer_s16_224_ni3(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=3,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def vanillamixer_s16_224_ni4(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=4,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def vanillamixer_b32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model

@register_model
def vanillamixer_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def vanillamixer_l32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model

@register_model
def vanillamixer_l16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024,
                      mlp_layer=vMlp, block_layer=vMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model



# parallelized Mixer models
@register_model
def paramixer_s32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                      **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def paramixer_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def paramixer_s16_224_ni2(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=2,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def paramixer_s16_224_ni3(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=3,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def paramixer_s16_224_ni4(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=4,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def paramixer_b32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model

@register_model
def paramixer_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def paramixer_l32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model

@register_model
def paramixer_l16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024,
                      mlp_layer=vMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model



# parallelized Mixer with symmetric weights
@register_model
def symmixer_s32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                      **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def symmixer_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def symmixer_s16_224_ni2(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=2,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def symmixer_s16_224_ni3(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=3,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def symmixer_s16_224_ni4(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=4,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def symmixer_b32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model

@register_model
def symmixer_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def symmixer_l32_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model

@register_model
def symmixer_l16_224(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024,
                      mlp_layer=sMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1,
                       **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model



# symmixer in symmetry breaking phase
@register_model
def asymmixer_s16_224_l00(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=0.,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def asymmixer_s16_224_l01(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=0.1,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def asymmixer_s16_224_l02(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=1e-2,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def asymmixer_s16_224_l03(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=1e-3,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def asymmixer_s16_224_l035(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=5e-4,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def asymmixer_s16_224_l04(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=1e-4,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def asymmixer_s16_224_l05(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=1e-5,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def asymmixer_s16_224_l06(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=1e-6,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

@register_model
def asymmixer_s16_224_l10(pretrained=False, **kwargs) -> MlpMixer:
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512,
                      mlp_layer=aMlp, block_layer=pMixerBlock,
                      hidden_ratio=1, n_iter=1, lcoeff=1.0,
                       **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model
