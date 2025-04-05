from dinotool.model import DinoFeatureExtractor, PCAModule
from dinotool.data import Video, FrameData
from PIL import Image
import numpy as np


class BatchHandler:
    def __init__(
        self, video: Video, feature_extractor: DinoFeatureExtractor, pca: PCAModule
    ):
        self.video = video
        self.feature_extractor = feature_extractor
        self.pca = pca

    def __call__(self, batch):
        features = self.feature_extractor(batch["img"])
        pca_features = self.pca.transform(features, flattened=False)

        framedata_list = []
        for batch_idx, frame_idx in enumerate(batch["frame_idx"].numpy()):
            img_frame = self.video[frame_idx]
            features_frame = features[batch_idx].unsqueeze(0)
            features_frame = self.feature_extractor.reshape_features(features_frame)[0]
            pca_frame = pca_features[batch_idx]

            framedata = FrameData(
                img=img_frame,
                features=features_frame,
                pca=pca_frame,
                frame_idx=frame_idx,
                flattened=False,
            )

            framedata_list.append(framedata)
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
