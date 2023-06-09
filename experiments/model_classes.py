import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, T5EncoderModel

def tensor_masking(tensor: Tensor, mask: Tensor, value: float = 0.0) -> Tensor:
    return tensor.masked_fill((~(mask.bool())).unsqueeze(-1), value)

class GlobalMaskedPooling(nn.Module):

    POOLING_TYPES = ("mean", "max")

    def __init__(
        self,
        pooling_type: str = "mean",
        dim: int = 1,
        normalize: bool = False,
        length_scaling: bool = False,
        scaling_square_root: bool = False,
        embedding_masking: bool = True,
    ):
        super().__init__()

        if pooling_type not in self.POOLING_TYPES:
            raise ValueError(
                f"{pooling_type} - is unavailable type." f' Available types: {", ".join(self.POOLING_TYPES)}'
            )

        if dim < 0:
            raise ValueError("Dimension (dim parameter) must be greater than zero")

        self.pooling_type = pooling_type
        self.dim = dim

        self.normalize = normalize
        self.length_scaling = length_scaling
        self.scaling_square_root = scaling_square_root

        self.embedding_masking = embedding_masking

        if self.pooling_type == "max":
            self.mask_value = -float("inf")
        else:
            self.mask_value = 0.0

    def forward(self, tensor: Tensor, pad_mask: Tensor) -> Tensor:
        lengths = pad_mask.sum(self.dim).float()

        if self.embedding_masking:
            tensor = tensor_masking(tensor, pad_mask, value=self.mask_value)

        if self.pooling_type == "mean":
            scaling = tensor.size(self.dim) / lengths
        else:
            scaling = torch.ones(tensor.size(0), device=tensor.device)

        if self.length_scaling:
            lengths_factor = lengths
            if self.scaling_square_root:
                lengths_factor = lengths_factor**0.5
            scaling /= lengths_factor

        scaling = scaling.masked_fill(lengths == 0, 1.0).unsqueeze(-1)

        if self.pooling_type == "mean":
            tensor = tensor.mean(self.dim)
        else:
            tensor, _ = tensor.max(self.dim)

        tensor *= scaling

        if self.normalize:
            tensor = F.normalize(tensor)

        return tensor

    def extra_repr(self) -> str:

        description = [
            f'pooling_type="{self.pooling_type}"',
            f"normalize={self.normalize}",
            f"length_scaling={self.length_scaling}",
            f"scaling_square_root={self.scaling_square_root}",
        ]

        description_message = ",\n".join(description)

        return
    
    
class ResidualLayer(nn.Module):
    
    def __init__(self, in_features, hidden_features):

        # Вызываем __init__ родителя - torch.nn.Module
        super().__init__()
        
        # определяем слои и все что захотим сохранять/использовать
        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_features)
        self.relu_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(in_features=hidden_features, out_features=in_features)
        self.relu_2 = torch.nn.ReLU()

    def forward(self, x):
        residual = x.clone()
        
        x = self.relu_1(self.linear_1(x))
        x = self.relu_2(self.linear_2(x))
        x = x + residual

        return x
    
class Classifier(nn.Module):
    def __init__(self, lang_model, pooling, dropout, n_classes):
        super().__init__()
        self.lang_model = lang_model
        self.device = lang_model.device
        self.pooling = pooling
        self.batch_norm = nn.BatchNorm1d(lang_model.config.d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # self.res_layer = ResidualLayer(lang_model.config.d_model, lang_model.config.d_model // 2)
        self.linear = nn.Linear(lang_model.config.d_model, n_classes)

    def forward(self, X, attention):
        X = self.lang_model(**X)
        X = self.pooling(X.last_hidden_state, attention)
        X = self.dropout(X)
        # X = self.batch_norm(X)
        # X = self.relu(X)
        # X = self.res_layer(X)
        X = self.linear(X)
        return X