from dinotool.model import DinoFeatureExtractor, PCAModule
from dinotool.data import Video, FrameData, LocalFeatures
from PIL import Image
import numpy as np
from typing import Union, List


class BatchHandler:
    def __init__(
        self, video: Video, feature_extractor: DinoFeatureExtractor, pca: Union[PCAModule, None] = None,
        progress_bar=None
    ):
        self.video = video
        self.feature_extractor = feature_extractor
        self.pca = pca
        self.progress_bar = progress_bar

    def __call__(self, batch):
        features = self.feature_extractor(batch["img"])
        if self.pca is not None:
            pca_features = self.pca.transform(features.flat().tensor, flattened=False)

        framedata_list = []
        for batch_idx, frame_idx in enumerate(batch["frame_idx"].numpy()):
            img_frame = self.video[frame_idx]
            feature_frame = features[batch_idx].full()

            if self.pca is not None:
                pca_frame = pca_features[batch_idx]
            else:
                pca_frame = None

            framedata = FrameData(
                img=img_frame,
                features=feature_frame,
                pca=pca_frame,
                frame_idx=int(frame_idx),
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
