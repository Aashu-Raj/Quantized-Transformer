import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import sys
import os

# Add path to access FQ-ViT modules
sys.path.append('../models')
from models.ptq import QAct, QLinear, QIntLayerNorm, QIntSoftmax
from config_mec import MECTransformerConfig


class QuantizedPositionalEncoding(nn.Module):
    """Quantized positional encoding layer"""
    
    def __init__(self, sequence_length, embed_dim, quant=False, calibrate=False, cfg=None):
        super(QuantizedPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.pos_encoding = self.positional_encoding(sequence_length, embed_dim)
        
        # Quantization for positional encoding addition
        self.qact_pos = QAct(quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_A,
                           calibration_mode=cfg.CALIBRATION_MODE_A,
                           observer_str=cfg.OBSERVER_A,
                           quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_out = QAct(quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_A,
                           calibration_mode=cfg.CALIBRATION_MODE_A,
                           observer_str=cfg.OBSERVER_A,
                           quantizer_str=cfg.QUANTIZER_A)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return torch.tensor(pos_encoding, dtype=torch.float32)

    def get_angles(self, pos, i, d_model):
        return pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        pos_enc = self.qact_pos(pos_enc)
        out = x + pos_enc
        return self.qact_out(out)


class QuantizedMultiHeadAttention(nn.Module):
    """Quantized Multi-Head Attention with Log-Int-Softmax"""
    
    def __init__(self, embed_dim, num_heads, quant=False, calibrate=False, cfg=None):
        super(QuantizedMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Quantized QKV projection
        self.qkv = QLinear(embed_dim, embed_dim * 3, bias=True,
                          quant=quant, calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_W,
                          calibration_mode=cfg.CALIBRATION_MODE_W,
                          observer_str=cfg.OBSERVER_W,
                          quantizer_str=cfg.QUANTIZER_W)
        
        # Quantized output projection
        self.out_proj = QLinear(embed_dim, embed_dim,
                               quant=quant, calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_W,
                               calibration_mode=cfg.CALIBRATION_MODE_W,
                               observer_str=cfg.OBSERVER_W,
                               quantizer_str=cfg.QUANTIZER_W)
        
        # Quantized activations
        self.qact_qkv = QAct(quant=quant, calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_A,
                           calibration_mode=cfg.CALIBRATION_MODE_A,
                           observer_str=cfg.OBSERVER_A,
                           quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_attn = QAct(quant=quant, calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_out = QAct(quant=quant, calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_A,
                           calibration_mode=cfg.CALIBRATION_MODE_A,
                           observer_str=cfg.OBSERVER_A,
                           quantizer_str=cfg.QUANTIZER_A)
        
        # Log-Int-Softmax for attention
        self.log_int_softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant, calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S)

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = self.qact_qkv(qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.qact_attn(attn)
        
        # Apply mask if provided
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, -1e9)
        
        # Quantized softmax
        attn = self.log_int_softmax(attn, self.qact_attn.quantizer.scale)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.qact_out(x)
        
        return x


class QuantizedTransformerEncoderLayer(nn.Module):
    """Quantized Transformer Encoder Layer with PTF LayerNorm"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, 
                 quant=False, calibrate=False, cfg=None):
        super(QuantizedTransformerEncoderLayer, self).__init__()
        
        # Quantized multi-head attention
        self.attention = QuantizedMultiHeadAttention(embed_dim, num_heads, 
                                                   quant=quant, calibrate=calibrate, cfg=cfg)
        
        # Quantized feed-forward network
        self.ffn = nn.Sequential(
            QLinear(embed_dim, ff_dim,
                   quant=quant, calibrate=calibrate,
                   bit_type=cfg.BIT_TYPE_W,
                   calibration_mode=cfg.CALIBRATION_MODE_W,
                   observer_str=cfg.OBSERVER_W,
                   quantizer_str=cfg.QUANTIZER_W),
            nn.GELU(),
            QAct(quant=quant, calibrate=calibrate,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A),
            QLinear(ff_dim, embed_dim,
                   quant=quant, calibrate=calibrate,
                   bit_type=cfg.BIT_TYPE_W,
                   calibration_mode=cfg.CALIBRATION_MODE_W,
                   observer_str=cfg.OBSERVER_W,
                   quantizer_str=cfg.QUANTIZER_W)
        )
        
        # Integer LayerNorm with PTF
        if cfg.INT_NORM:
            self.layernorm1 = QIntLayerNorm(embed_dim)
            self.layernorm2 = QIntLayerNorm(embed_dim)
        else:
            self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Quantized activations for residual connections
        self.qact1 = QAct(quant=quant, calibrate=calibrate,
                         bit_type=cfg.BIT_TYPE_A,
                         calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                         observer_str=cfg.OBSERVER_A_LN,
                         quantizer_str=cfg.QUANTIZER_A_LN)
        
        self.qact2 = QAct(quant=quant, calibrate=calibrate,
                         bit_type=cfg.BIT_TYPE_A,
                         calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                         observer_str=cfg.OBSERVER_A_LN,
                         quantizer_str=cfg.QUANTIZER_A_LN)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None, last_quantizer=None):
        # Self-attention block with residual connection
        if hasattr(self.layernorm1, 'mode'):  # QIntLayerNorm
            attn_out = self.attention(
                self.layernorm1(x, last_quantizer, self.qact1.quantizer), 
                attention_mask
            )
        else:  # Regular LayerNorm
            attn_out = self.attention(self.layernorm1(x), attention_mask)
        
        attn_out = self.dropout1(attn_out)
        x = self.qact1(x + attn_out)
        
        # Feed-forward block with residual connection
        if hasattr(self.layernorm2, 'mode'):  # QIntLayerNorm
            ffn_out = self.ffn(self.layernorm2(x, self.qact1.quantizer, self.qact2.quantizer))
        else:  # Regular LayerNorm
            ffn_out = self.ffn(self.layernorm2(x))
        
        ffn_out = self.dropout2(ffn_out)
        x = self.qact2(x + ffn_out)
        
        return x


class QuantizedTransformerEncoder(nn.Module):
    """Quantized Transformer Encoder for MEC resource allocation"""
    
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, 
                 dropout_rate, output_dim=2, max_seq_len=100,
                 quant=False, calibrate=False, cfg=None):
        super(QuantizedTransformerEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.cfg = cfg
        
        # Quantized input projection
        self.input_proj = QLinear(embed_dim, embed_dim,
                                 quant=quant, calibrate=calibrate,
                                 bit_type=cfg.BIT_TYPE_W,
                                 calibration_mode=cfg.CALIBRATION_MODE_W,
                                 observer_str=cfg.OBSERVER_W,
                                 quantizer_str=cfg.QUANTIZER_W)
        
        # Quantized positional encoding
        self.pos_encoding = QuantizedPositionalEncoding(max_seq_len, embed_dim,
                                                       quant=quant, calibrate=calibrate, cfg=cfg)
        
        # Stack of quantized encoder layers
        self.enc_layers = nn.ModuleList([
            QuantizedTransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate,
                                           quant=quant, calibrate=calibrate, cfg=cfg)
            for _ in range(num_layers)
        ])
        
        # Quantized output layer
        self.output_layer = QLinear(embed_dim, output_dim,
                                   quant=quant, calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_W,
                                   calibration_mode=cfg.CALIBRATION_MODE_W,
                                   observer_str=cfg.OBSERVER_W,
                                   quantizer_str=cfg.QUANTIZER_W)
        
        self.qact_input = QAct(quant=quant, calibrate=calibrate,
                              bit_type=cfg.BIT_TYPE_A,
                              calibration_mode=cfg.CALIBRATION_MODE_A,
                              observer_str=cfg.OBSERVER_A,
                              quantizer_str=cfg.QUANTIZER_A)
        
        self.qact_output = QAct(quant=quant, calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)

    def model_quant(self):
        """Enable quantization for all layers"""
        for m in self.modules():
            if type(m) in [QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        """Disable quantization for all layers"""
        for m in self.modules():
            if type(m) in [QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        """Start calibration mode"""
        for m in self.modules():
            if type(m) in [QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        """Final calibration step"""
        for m in self.modules():
            if type(m) in [QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        """End calibration mode"""
        for m in self.modules():
            if type(m) in [QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward(self, inputs, attention_mask=None):
        x = self.input_proj(inputs)
        x = self.qact_input(x)
        x = self.pos_encoding(x)
        
        last_quantizer = None
        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, attention_mask, last_quantizer)
            if i < len(self.enc_layers) - 1:
                last_quantizer = enc_layer.qact2.quantizer
        
        x = self.output_layer(x)
        x = torch.sigmoid(self.qact_output(x))
        
        return x


class QuantizedMemoryDNN:
    """Quantized Memory DNN for MEC resource allocation with transformer backend"""
    
    def __init__(self, net, max_users=None, learning_rate=0.001, memory_size=2000,
                 batch_size=128, training_interval=10, output_dim=2, 
                 quant=True, calibrate=False, cfg=None):
        
        self.net = net
        self.max_users = max_users if max_users is not None else net[1]
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.training_interval = training_interval
        self.kernal_size = int(net[0] / self.max_users)
        self.output_dim = output_dim
        self.quant = quant
        self.calibrate = calibrate
        
        # Initialize configuration if not provided
        if cfg is None:
            cfg = MECTransformerConfig()
        self.cfg = cfg
        
        # Build quantized transformer model
        self.model = QuantizedTransformerEncoder(
            num_layers=4,
            embed_dim=self.kernal_size,  # Use kernel size as embed dim
            num_heads=4,
            ff_dim=256,
            dropout_rate=0.1,
            output_dim=self.output_dim,
            max_seq_len=self.max_users,
            quant=quant,
            calibrate=calibrate,
            cfg=cfg
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        self.memory_counter = 0
        self.cost_his = []

    def remember(self, x, y):
        """Store experience in memory"""
        self.memory.append((x, y))
        self.memory_counter += 1

    def learn(self):
        """Train the quantized model"""
        if self.memory_counter < self.batch_size:
            return
            
        mini_batch = random.sample(self.memory, self.batch_size)
        h_train = np.array([data[0] for data in mini_batch])
        y_train = np.array([data[1] for data in mini_batch])
        
        # Reshape input data
        h_train = h_train.reshape(-1, self.max_users, self.kernal_size)
        
        # Convert to tensors
        h_train = torch.tensor(h_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(h_train)
        loss = self.criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.cost_his.append(loss.item())

    def encode(self, x, y):
        """Encode experience and train if needed"""
        self.remember(x, y)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def decode(self, x, K=None, decoder_mode='OP'):
        """Decode decision using quantized model"""
        if K is None:
            K = self.max_users
            
        # Reshape input
        x_reshaped = x.reshape(1, self.max_users, self.kernal_size)
        x_tensor = torch.tensor(x_reshaped, dtype=torch.float32).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            y_pred = self.model(x_tensor).cpu().numpy()
        
        y_pred = y_pred[0]
        
        # Convert to binary decisions
        m_pred = np.zeros_like(y_pred)
        m_pred[np.arange(self.max_users), y_pred.argmax(axis=1)] = 1
        
        m_list = [m_pred]
        return m_pred, m_list

    def start_calibration(self, calibration_data):
        """Calibrate the quantized model"""
        print("Starting quantization calibration...")
        self.model.model_open_calibrate()
        
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                x_reshaped = data.reshape(1, self.max_users, self.kernal_size)
                x_tensor = torch.tensor(x_reshaped, dtype=torch.float32).to(self.device)
                
                if i == len(calibration_data) - 1:
                    self.model.model_open_last_calibrate()
                
                _ = self.model(x_tensor)
        
        self.model.model_close_calibrate()
        self.model.model_quant()
        print("Quantization calibration completed!")

    def plot_cost(self):
        """Plot training cost"""
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.title('Quantized Transformer MemoryDNN Training Cost')
        plt.grid(True)
        plt.show()

    def save_model(self, path):
        """Save the quantized model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
            'cost_history': self.cost_his
        }, path)

    def load_model(self, path):
        """Load the quantized model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cost_his = checkpoint['cost_history'] 