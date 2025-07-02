import sys
import os
sys.path.append('../models')
from models.ptq.bit_type import BIT_TYPE_DICT


class MECTransformerConfig:
    """Configuration for quantized MEC Transformer models"""
    
    def __init__(self, ptf=True, lis=True, quant_method='minmax', 
                 bits_w=8, bits_a=8, bits_s=4):
        '''
        Configuration for MEC transformer quantization
        ptf: Power-of-Two Factor for LayerNorm quantization
        lis: Log-Int-Softmax for attention quantization
        quant_method: Quantization method for calibration
        '''
        
        # Bit precision settings
        if bits_w == 8:
            self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        elif bits_w == 4:
            self.BIT_TYPE_W = BIT_TYPE_DICT['uint4']
            
        if bits_a == 8:
            self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        elif bits_a == 4:
            self.BIT_TYPE_A = BIT_TYPE_DICT['uint4']
            
        if bits_s == 4:
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
        elif bits_s == 8:
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']

        # Observer settings for calibration
        self.OBSERVER_W = 'minmax'  # Weight observer
        self.OBSERVER_A = quant_method  # Activation observer

        # Quantizer settings
        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'

        # Calibration modes
        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'

        # Enable specialized quantization techniques
        if lis:
            self.INT_SOFTMAX = True
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
            
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A 