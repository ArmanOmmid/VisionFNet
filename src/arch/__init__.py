
import sys
import os

repository_root = '/'.join(__file__.split('/')[:__file__.split('/').index('src')])
sys.path.append(repository_root)

from .archived import basic_fcn, customfcn1, customfcn2, transfer_fcn, transfer_unet, unet
from .vit import vit_token, vit
from .fvit import fvit, fvit_gft, fvit_fno, fvit_fsa, fvit_cross, fvit_spectral, fvit_spectral_classtoken, fvit_spectral_multiscale, fvit_token, fvit_mgft, fvit_monolith
from .dsc import dsc_arch
