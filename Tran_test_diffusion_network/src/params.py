
import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=128,
    learning_rate=0.001,
    max_grad_norm=None,

    # Data params
    sample_rate=55,
    #n_mels set to the npy size
    n_mels=20,
    n_fft=256,
    hop_samples=16,
    # frame number at once
    crop_mel_frames=104,

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    noise_schedule=np.linspace(1e-4, 0.035, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35],
)
