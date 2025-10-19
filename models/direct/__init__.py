"""Direct diffusion models.

This subpackage currently exposes a spatio-temporal UNet that is able to
adapt to a varying number of input channels at runtime.  The adapter was
introduced to prevent crashes when the dataloader returns a slightly
different number of conditioning channels than the configuration used
when instantiating the model.  Such situations previously triggered a
``RuntimeError`` inside the first convolution layer.
"""

from .unet import DirectSpatioTemporalUNet

__all__ = ["DirectSpatioTemporalUNet"]
