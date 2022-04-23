import torch
import math

def compute_conv_transpose(w, s, k):
  return math.floor((w - 1) * s + k)

def compute_conv(w, s, k):
  return math.floor((w - k) / s + 1)

def find_parameters(in_size, out_size, n_layer, layer = 'conv', s = [1, 2], k = [3, 4]):
  '''
    s: tensor of strides
    k: tensor of kernels
    Return: (strides, kernels)
  '''
  s = torch.tensor(s)
  k = torch.tensor(k)
  count = 0
  if layer == 'conv':
    fnc = compute_conv
  elif layer == 'conv_transpose':
    fnc = compute_conv_transpose
  while count < 1000:
    s_c = s[torch.randint(0, len(s), (n_layer, ))]
    k_c = k[torch.randint(0, len(k), (n_layer, ))]

    fi = in_size
    for (si, ki) in zip(s_c, k_c):
      fi = fnc(fi, si, ki)
    if fi == out_size:
      return s_c, k_c.data
    count += 1
  return None

import os
from omegaconf import OmegaConf

def load_seq_config(config_dir, config_name):
    cf = OmegaConf.load(os.path.join(config_dir, config_name  +'.yaml'))
    defaults = []
    sub_module = cf.get('defaults', [])
    for sm in sub_module:
        name, sub = sm.items()[0]
        cf_sub_module = load_seq_config(os.path.join(config_dir, name), sub)
        cf_item = OmegaConf.create({})
        OmegaConf.update(cf_item, name, cf_sub_module)
        defaults.append(cf_item)
    cf = OmegaConf.merge(cf.copy(), *defaults)
    return cf
