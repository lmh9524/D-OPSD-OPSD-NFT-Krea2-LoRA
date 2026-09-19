"""Runnable experiment configurations."""

from dflow.config.recipe.flux2_ref2img_lora import Flux2Ref2ImgLoRARecipe
from dflow.config.recipe.flux2_t2i_flowgrpo import Flux2T2IFlowGRPORecipe
from dflow.config.recipe.flux2_t2i_lora import Flux2T2ILoRARecipe
from dflow.config.recipe.krea2_dopsd_lora import Krea2DOPSDRecipe
from dflow.config.recipe.krea2_opsd_nft_lora import Krea2OPSDNFTRecipe
from dflow.config.recipe.krea2_ref2img_lora import Krea2Ref2ImgLoRARecipe

__all__ = [
    "Flux2Ref2ImgLoRARecipe",
    "Flux2T2IFlowGRPORecipe",
    "Flux2T2ILoRARecipe",
    "Krea2DOPSDRecipe",
    "Krea2OPSDNFTRecipe",
    "Krea2Ref2ImgLoRARecipe",
]
