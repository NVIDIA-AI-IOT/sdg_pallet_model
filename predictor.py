import torch
from clustering import Clustering
import utils
import numpy as np
import cv2


class Predictor:

    def __init__(self, engine_path: str, inference_sizes=[256], threshold=0.8, eps=0.00125):
        self.engine = utils.load_trt_engine_wrapper(
            engine_path,
            input_names=["input"],
            output_names=["heatmap", "vectormap"]
        )
        self.inference_sizes = inference_sizes
        self.clustering = Clustering(eps=eps)
        self.offset_grids = [
            utils.make_offset_grid((size, size)).to("cuda")
            for size in self.inference_sizes
        ]
        self.peak_threshold = threshold
        self.peak_window = (15, 31)

    @torch.no_grad()
    def predict(self, image):

        h, w = image.shape[0], image.shape[1]

        all_keypoints = []

        for size, grid in zip(self.inference_sizes, self.offset_grids):

            if w >= h:
                iw = size
                scale = iw / w
                ih = int(h * scale)
            else:
                ih = size
                scale = ih / h
                iw = int(w * scale)

            image_i = cv2.resize(image, (iw, ih))

            x_i_img = utils.format_bgr8_image(image_i).cuda()

            x_i = torch.zeros((1, 3, size, size), dtype=x_i_img.dtype, device=x_i_img.device)

            x_i[0, :, :ih, :iw] = x_i_img

            heatmap, vectormap = self.engine(x_i)
            heatmap = torch.sigmoid(heatmap)
                    
            keypointmap = utils.vectormap_to_keypointmap(
                grid,
                vectormap
            )
            
            peak_mask = utils.find_heatmap_peak_mask(
                heatmap, 
                self.peak_window,
                self.peak_threshold
            )
            
            keypoints = keypointmap[0][peak_mask[0, 0]]

            all_keypoints.append(keypoints / scale)
        
        keypoints = torch.cat(all_keypoints, dim=0)
        keypoints = self.clustering.cluster(keypoints)

        return keypoints.detach().cpu().numpy()