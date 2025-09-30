import torch
from sklearn.decomposition import PCA
from torch import nn
from einops import rearrange
from typing import List, Tuple
import warnings
from torchvision import transforms

from transformers import AutoImageProcessor, AutoModel

from dinotool.data import calculate_dino_dimensions, LocalFeatures


def load_model(model_name: str = "dinov2_vits14_reg") -> nn.Module:
    """Load a model for DINO or OpenCLIP.
    This function loads a DINO or OpenCLIP model based on the provided model name.
    If the model name starts with "hf-hub:timm", it is assumed to be an OpenCLIP model.
    Otherwise, it is assumed to be a DINO model.
    Args:
        model_name (str): name of the model to load.
    Returns:
        nn.Module: Model.
    """
    if model_name.startswith("hf-hub:timm"):
        from open_clip import create_model_from_pretrained

        model = create_model_from_pretrained(model_name, return_transform=False)
        try:
            patch_size = model.visual.patch_size[0]
        except AttributeError:
            patch_size = model.visual.trunk.patch_embed.proj.kernel_size[0]
        model.patch_size = patch_size
    
    elif model_name.startswith("facebook/dinov3"):
        model = AutoModel.from_pretrained(model_name)
        model.patch_size = model.config.patch_size
    
    elif model_name.startswith("NVlabs/RADIO/"):
        model_version = model_name.split("/")[-1]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="warmup, rep, and use_cuda_graph parameters are deprecated.*",
            )
            model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version)

    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is available*")
            warnings.filterwarnings(
                "ignore",
                message="warmup, rep, and use_cuda_graph parameters are deprecated.*",
            )
            model = torch.hub.load("facebookresearch/dinov2", model_name)

    return model


class OpenCLIPFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """Feature extractor for OpenCLIP model.
        Args:
            model (nn.Module): OpenCLIP model.
            input_size (Tuple[int, int]): feature map size (width, height).
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

        self.patch_size = model.patch_size

    def forward(
        self,
        batch: torch.Tensor,
        flattened=True,
        normalized=True,
        return_clstoken=False,
    ):
        if return_clstoken:
            with torch.no_grad():
                batch = batch.to(self.device)
                features = self.model.encode_image(batch, normalize=normalized)
                return features

        b, c, h, w = batch.shape
        dims = calculate_dino_dimensions((w, h), self.patch_size)
        h_featmap, w_featmap = dims["h_featmap"], dims["w_featmap"]

        with torch.no_grad():
            batch = batch.to(self.device)
            feature_tensor = self.model.forward_intermediates(batch)[
                "image_intermediates"
            ][0]

        reshaped_tensor = rearrange(
            feature_tensor, "b f h w -> b h w f", h=h_featmap, w=w_featmap
        )

        features = LocalFeatures(
            reshaped_tensor, is_flattened=False, h=h_featmap, w=w_featmap
        )

        if normalized:
            features = features.normalize()

        if flattened:
            features = features.flat()
        else:
            features = features.full()

        return features


class DinoFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, device: str = "cuda"):
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

    def forward(
        self,
        batch: torch.Tensor,
        flattened=True,
        normalized=True,
        return_clstoken=False,
    ):
        if return_clstoken:
            with torch.no_grad():
                batch = batch.to(self.device)
                features = self.model.forward(batch)
                if normalized:
                    features = torch.nn.functional.normalize(features, dim=-1)
                return features

        b, c, h, w = batch.shape
        dims = calculate_dino_dimensions((w, h), self.patch_size)
        h_featmap, w_featmap = dims["h_featmap"], dims["w_featmap"]

        with torch.no_grad():
            batch = batch.to(self.device)
            feature_tensor = self.model.forward_features(batch)["x_norm_patchtokens"]

        features = LocalFeatures(
            feature_tensor, is_flattened=True, h=h_featmap, w=w_featmap
        )
        if normalized:
            features = features.normalize()

        if flattened:
            features = features.flat()
        else:
            features = features.full()

        return features

class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """Feature extractor for DINOv3 model.
        Args:
            model (nn.Module): DINOv3 model.
            input_size (Tuple[int, int]): feature map size (width, height).
            device (str): device to use for computation.
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

        self.patch_size = model.patch_size

    def forward(
        self,
        batch: torch.Tensor,
        flattened=True,
        normalized=True,
        return_clstoken=False,
    ):

        b, c, h, w = batch.shape
        dims = calculate_dino_dimensions((w, h), self.patch_size)
        h_featmap, w_featmap = dims["h_featmap"], dims["w_featmap"]

        with torch.inference_mode():
            batch = batch.to(self.device)
            outputs = self.model(batch)

        if return_clstoken:
            features = outputs.pooler_output
            if normalized:
                features = torch.nn.functional.normalize(features, dim=-1)
            return features
        
        feature_tensor = outputs.last_hidden_state[:,5:,:]

        features = LocalFeatures(
            feature_tensor, is_flattened=True, h=h_featmap, w=w_featmap
        )

        if normalized:
            features = features.normalize()

        if flattened:
            features = features.flat()
        else:
            features = features.full()

        return features

class RADIOFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """Feature extractor for RADIO model.
        Args:
            model (nn.Module): RADIO model.
            input_size (Tuple[int, int]): feature map size (width, height).
            device (str): device to use for computation.
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

        self.patch_size = model.patch_size

    def forward(
        self,
        batch: torch.Tensor,
        flattened=True,
        normalized=True,
        return_clstoken=False,
    ):

        b, c, h, w = batch.shape
        dims = calculate_dino_dimensions((w, h), self.patch_size)
        h_featmap, w_featmap = dims["h_featmap"], dims["w_featmap"]

        with torch.no_grad():
            batch = batch.to(self.device)
            global_feature, feature_tensor = self.model(batch, feature_fmt='NCHW')

        if return_clstoken:
            if normalized:
                global_feature = torch.nn.functional.normalize(global_feature, dim=-1)
            return global_feature
        
        feature_tensor = rearrange(
            feature_tensor, "b f h w -> b h w f",
        )
        
        features = LocalFeatures(
            feature_tensor, is_flattened=False,
            h=h_featmap, w=w_featmap
        )

        if normalized:
            features = features.normalize()

        if flattened:
            features = features.flat()
        else:
            features = features.full()

        return features


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

    def fit(self, features: torch.Tensor, verbose: bool = True):
        features = self.__check_features(features)
        b, hw, f = features.shape
        self.pca.fit(features.reshape(b * hw, f))
        if verbose:
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
                for i in range(self.n_components):
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
