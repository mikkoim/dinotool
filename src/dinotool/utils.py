from dinotool.model import DinoFeatureExtractor, PCAModule
from dinotool.data import Video, FrameData, LocalFeatures, ImageDirectory
from PIL import Image
import numpy as np
from typing import Union, List


class BatchHandler:
    def __init__(
        self,
        source: Union[Video, ImageDirectory],
        feature_extractor: DinoFeatureExtractor,
        pca: Union[PCAModule, None] = None,
        progress_bar=None,
    ):
        self.source = source
        self.feature_extractor = feature_extractor
        self.pca = pca
        self.progress_bar = progress_bar

    def __call__(self, batch):
        features = self.feature_extractor(batch["img"])
        if self.pca is not None:
            pca_features = self.pca.transform(features.flat().tensor, flattened=False)

        framedata_list = []
        if "filename" in batch:
            identifiers = batch["filename"]
            identifier_type = "filename"
        elif "frame_idx" in batch:
            identifiers = batch["frame_idx"].numpy()
            identifier_type = "frame_idx"
        else:
            raise ValueError("Batch must contain either 'filename' or 'frame_idx'.")

        for batch_idx, identifier in enumerate(identifiers):
            if identifier_type == "filename":
                img_source_data = self.source.get_by_name(identifier)
                frame_data_kwargs = {"filename": identifier}
            else:
                img_source_data = self.source[identifier]
                frame_data_kwargs = {"frame_idx": int(identifier)}

            feature_frame = features[batch_idx].full()

            if self.pca is not None:
                pca_frame = pca_features[batch_idx]
            else:
                pca_frame = None

            framedata = FrameData(
                img=img_source_data,
                features=feature_frame,
                pca=pca_frame,
                **frame_data_kwargs,
            )

            framedata_list.append(framedata)
            if self.progress_bar is not None:
                self.progress_bar.update(1)
        return framedata_list


def frame_visualizer(frame_data: FrameData, output_size=(480, 270), only_pca=False):
    pca_img = Image.fromarray((frame_data.pca * 255).astype(np.uint8)).resize(
        output_size, Image.NEAREST
    )
    if only_pca:
        return pca_img
    resized_img = frame_data.img.resize(output_size, Image.LANCZOS)

    stacked = np.vstack([np.array(resized_img), np.array(pca_img)])
    out_img = Image.fromarray(stacked)
    return out_img
