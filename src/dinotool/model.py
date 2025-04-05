import torch
from sklearn.decomposition import PCA
from torch import nn
from einops import rearrange
from typing import List, Tuple
from einops import rearrange
import warnings

from dinotool.data import calculate_dino_dimensions


def load_dino_model(model_name: str = "dinov2_vits14_reg") -> nn.Module:
    """Load a DINO model from the facebookresearch/dinov2 repository.
    Args:
        model_name (str): name of the DINO model to load.
    Returns:
        nn.Module: DINO model.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xFormers is available*")
        warnings.filterwarnings(
            "ignore",
            message="warmup, rep, and use_cuda_graph parameters are deprecated.*",
        )
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    return model


class DinoFeatureExtractor(nn.Module):
    def __init__(
        self, model: nn.Module, input_size: Tuple[int, int] = None, device: str = "cuda"
    ):
        """Feature extractor for DINO model.
        Args:
            model (nn.Module): DINO model.
            input_size (Tuple[int, int]): feature map size (width, height).
            device (str): device to use for computation.
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

        self.patch_size = model.patch_size
        self.feat_dim = model.num_features
        self.input_size = input_size

        if input_size is not None:
            dino_dims = calculate_dino_dimensions(
                input_size, patch_size=self.patch_size
            )
            self.w_featmap = dino_dims["w_featmap"]
            self.h_featmap = dino_dims["h_featmap"]

    def forward(self, batch: torch.Tensor, flattened=True, normalized=True):
        b, c, h, w = batch.shape
        with torch.no_grad():
            batch = batch.to(self.device)
            features = self.model.forward_features(batch)["x_norm_patchtokens"]
        b, hw, f = features.shape
        if normalized:
            features = nn.functional.normalize(
                rearrange(features, "b hw f -> (b hw) f"), dim=1
            )
            features = features.reshape(b, hw, f)
        if flattened:
            return features
        features = self.reshape_features(features)
        return features

    def reshape_features(self, features: torch.Tensor):
        """Reshape features to (b, h, w, f) format.
        Args:
            features (torch.Tensor): features to reshape.
        Returns:
            torch.Tensor: reshaped features.
        """
        if self.input_size is None:
            raise ValueError("input_size must be set when reshaping features")
        b, hw, f = features.shape
        return features.reshape(b, self.h_featmap, self.w_featmap, f)


class PCAModule:
    def __init__(self, n_components: int = 3, feature_map_size: Tuple[int, int] = None):
        """PCA module for DINO features.
        Args:
            n_components (int): number of PCA components.
            feature_map_size (Tuple[int, int]): feature map size (width, height).
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.feature_map_size = feature_map_size

    def __check_features(self, features: torch.Tensor):
        if features.ndim != 3:
            raise ValueError("features must be 3D tensor of form (b, hw, f)")
        if features.device != "cpu":
            features = features.cpu()
        return features

    def fit(self, features: torch.Tensor):
        features = self.__check_features(features)
        b, hw, f = features.shape
        self.pca.fit(features.reshape(b * hw, f))
        print(f"Fitted PCA with {self.pca.n_components_} components")
        if self.n_components < 8:
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")

    def transform(self, features, flattened=True, normalized=True):
        features = self.__check_features(features)
        b, hw, f = features.shape
        pca_features = self.pca.transform(features.reshape(b * hw, f))
        pca_features = pca_features.reshape(b, hw, self.n_components)
        if normalized:
            for bi in range(b):
                for i in range(3):
                    # min_max scaling
                    pca_features[bi, :, i] = (
                        pca_features[bi, :, i] - pca_features[bi, :, i].min()
                    ) / (pca_features[bi, :, i].max() - pca_features[bi, :, i].min())
        if flattened:
            return pca_features
        if self.feature_map_size is None:
            raise ValueError("feature_map_size must be set when flattened=False")
        pca_features = pca_features.reshape(
            b, self.feature_map_size[1], self.feature_map_size[0], self.n_components
        )
        return pca_features

    def set_feature_map_size(self, feature_map_size: Tuple[int, int]):
        """Set the feature map size.
        Args:
            feature_map_size (Tuple[int, int]): feature map size (width, height).
        """
        self.feature_map_size = feature_map_size
