import logging
import os

import numpy as np
import SimpleITK as sitk
import torch
from monai.transforms import BorderPad, Compose, Lambda, ScaleIntensity, SpatialCrop

from .landmarks import LABEL_GROUPES
from .markups import (
    agent_index_zyx_to_lps_xyz,
    gen_control_points,
    volume_stem,
    write_merged_mrk_json,
    write_mrk_json,
)
from .output_names import LANDMARK_MRK_SUFFIX

logger = logging.getLogger(__name__)


class Environment:
    def __init__(
        self,
        patient_id,
        padding,
        device,
        correct_contrast=False,
        verbose=False,
    ) -> None:
        self.patient_id = patient_id
        self.padding = padding.astype(np.int16)
        self.device = device
        self.verbose = verbose
        # AddChannel 은 MONAI 1.3 에서 제거됨 — Lambda 로 채널 축 추가
        self.transform = Compose(
            [Lambda(lambda x: x[None]), BorderPad(spatial_border=self.padding.tolist())]
        )
        self.scale_nbr = 0
        self.available_lm = []
        self.data = {}
        self.predicted_landmarks = {}
        self._reference_images: dict = {}

    def LoadImages(self, images_path):
        for scale_id, path in images_path.items():
            data = {"path": path}
            img = sitk.ReadImage(path)
            self._reference_images[scale_id] = img
            img_ar = sitk.GetArrayFromImage(img)
            # MONAI 변환 결과가 MetaTensor 가 될 수 있어 as_tensor 사용
            data["image"] = torch.as_tensor(self.transform(img_ar)).type(torch.int16)
            data["spacing"] = np.array(img.GetSpacing())
            origin = img.GetOrigin()
            data["origin"] = np.array([origin[2], origin[1], origin[0]])
            data["size"] = np.array(np.shape(img_ar))
            data["landmarks"] = {}
            self.data[scale_id] = data
            self.scale_nbr += 1

    def get_reference_image(self, scale_key: str) -> sitk.Image:
        if scale_key in self._reference_images:
            return self._reference_images[scale_key]
        img_path = self.data[scale_key]["path"]
        img = sitk.ReadImage(img_path)
        self._reference_images[scale_key] = img
        return img

    def SavePredictedLandmarks(
        self,
        scale_key,
        out_path=None,
        *,
        save_grouped: bool = False,
        save_merged: bool = True,
        merged_suffix: str = LANDMARK_MRK_SUFFIX,
    ):
        logger.info(
            "Saving predicted landmarks for %s at scale %s",
            self.patient_id,
            scale_key,
        )

        ref = self.get_reference_image(scale_key)
        id_stem = volume_stem(self.patient_id)

        all_coords: dict[str, dict[str, float]] = {}
        landmark_dic = {}
        for landmark, pos in self.predicted_landmarks.items():
            x, y, z = agent_index_zyx_to_lps_xyz(ref, pos)
            all_coords[landmark] = {"x": x, "y": y, "z": z}
            g = LABEL_GROUPES[landmark]
            entry = {"label": landmark, "coord": [x, y, z]}
            if g in landmark_dic:
                landmark_dic[g].append(entry)
            else:
                landmark_dic[g] = [entry]

        base_dir = out_path if out_path is not None else os.path.dirname(
            self.data[scale_key]["path"]
        )

        if save_grouped:
            for group, lst in landmark_dic.items():
                json_name = f"{id_stem}_lm_Pred_{group}.mrk.json"
                file_path = os.path.join(base_dir, json_name)
                groupe_data = {
                    lm["label"]: {
                        "x": lm["coord"][0],
                        "y": lm["coord"][1],
                        "z": lm["coord"][2],
                    }
                    for lm in lst
                }
                write_mrk_json(gen_control_points(groupe_data), file_path)

        if save_merged and all_coords:
            merged_path = os.path.join(base_dir, f"{id_stem}{merged_suffix}")
            write_merged_mrk_json(all_coords, merged_path)

    def LandmarkIsPresent(self, landmark):
        return landmark in self.available_lm

    def GetLandmarkPos(self, scale, landmark):
        return self.data[scale]["landmarks"][landmark]

    def GetZone(self, scale, center, crop_size):
        crop_transform = SpatialCrop((center + self.padding).tolist(), crop_size)
        rescale = ScaleIntensity(minv=-1.0, maxv=1.0, factor=None)
        crop = crop_transform(self.data[scale]["image"])
        crop = rescale(crop).type(torch.float32)
        return crop

    def GetSpacing(self, scale):
        return self.data[scale]["spacing"]

    def GetSize(self, scale):
        return self.data[scale]["size"]

    def AddPredictedLandmark(self, lm_id, lm_pos):
        self.predicted_landmarks[lm_id] = lm_pos


def __getattr__(name: str):
    if name == "Environement":
        from .compat import deprecate

        deprecate("Environement", "Environment")
        return Environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
