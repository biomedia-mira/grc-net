import os
import os.path as path
import gc

from totalsegmentator.python_api import totalsegmentator
from anatomix.segmentation.segmentation_utils import load_model

import torch
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from monai.data import Dataset
from img.processing import one_hot_labelmap
from monai.transforms import Compose, Rand3DElasticd, ToTensord, EnsureTyped, Lambdad
from monai.transforms import EnsureChannelFirstd
from img.transforms import Resamplerd
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference

# remap the class labels in MM-WHS.
def remap_labels(label):
    remap_dict = {
        0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7,
    }
    for raw_val, class_idx in remap_dict.items():
        label[label == raw_val] = class_idx
    return label

def load_sitk_array(path):
    img = sitk.ReadImage(path)
    img = sitk.DICOMOrient(img, "RAS")
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    tensor = torch.from_numpy(arr)

    meta = {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
    }
    return MetaTensor(tensor, meta=meta)

def get_image_label_transforms(spacing, crop_size, normalizer):
    val_transforms = Compose(
        [
            Lambdad(keys=["image", "gt"], func=load_sitk_array, allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "gt"], channel_dim='no_channel', allow_missing_keys=True),
            EnsureTyped(keys=["image", "gt"], device='cpu', allow_missing_keys=True),
            EnsureTyped(keys=["image", "gt"], dtype=torch.float32, allow_missing_keys=True),
            Resamplerd(keys=["image", "gt"], out_spacing=spacing, out_size=crop_size, is_label=[False, True], allow_missing_keys=True),
            EnsureTyped(keys=["image", "gt"], dtype=torch.float32, allow_missing_keys=True),
            Lambdad(keys="gt", func=remap_labels, allow_missing_keys=True),
            normalizer,
        ]
    )
    return val_transforms

class ImageSegmentationOneHotDataset(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, 
                csv_file_img,
                num_classes, spatial_size, spacing,
                mode="Anatomix", csv_file_seg=None, normalizer=None, binarize=False, augmentation=False, label_smoothing=0.0):
        """
        Args:
        :param csv_file_img (string): Path to csv file with image filenames.
        :param csv_file_seg (string, optional): Path to csv file with segmentation filenames.
        :param normalizer_img (callable, optional): Optional transform to be applied on each image.
        :param normalizer_seg (callable, optional): Optional transform to be applied on each segmentation.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        is_labelled = csv_file_seg is not None and path.exists(csv_file_seg)
        self.img_data = pd.read_csv(csv_file_img, comment="#") if path.exists(csv_file_img) else []
        self.seg_data = pd.read_csv(csv_file_seg, comment="#") if is_labelled else []

        self.augmentation = augmentation

        self.augment = Compose([
            ToTensord(keys=["image", "labelmap"], allow_missing_keys=True),
            Rand3DElasticd(
                keys=["image", "labelmap"],
                mode=["bilinear", "nearest"],
                sigma_range=(5, 7),
                magnitude_range=(4, 4),
                spatial_size=None,
                translate_range=0,
                rotate_range=0,
                scale_range=0,
                padding_mode="border",
                allow_missing_keys=True
            ),
        ])

        self.samples = []

        if not is_labelled:
            model = load_model(
                pretrained_ckpt='anatomix/model-weights/anatomix.pth',
                n_classes=num_classes-1,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            model.load_state_dict(torch.load("saved_models/segmentation/anatomix_trained_MM-WHS.pth"))
            model.eval()

        for idx in tqdm(range(len(self.img_data)), desc='Loading Data'):
            img_path    = self.img_data.iloc[idx, 0]
            img_fname = os.path.basename(img_path)

            transforms  = get_image_label_transforms(spacing, spatial_size, normalizer)
            io_dict     = {"image": img_path}

            if is_labelled:
                # labelled: hand in the seg CSV path so the transform will load it
                io_dict["gt"] = self.seg_data.iloc[idx, 0]
                sample = transforms(io_dict)
                label_vol = sample["gt"].cpu().squeeze()
            else:
                # unlabelled: run a semi-supervised model on the preprocessed image
                if mode == "Anatomix":
                    sample   = transforms(io_dict)
                    img_t    = sample["image"].unsqueeze(0).to(device)

                    logits   = sliding_window_inference(
                        img_t,
                        roi_size=spatial_size,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.7,
                    )
                    label_vol = torch.argmax(logits, dim=1)[0].cpu()
                elif mode == "TotalSegmentator":
                    nib_img = nib.load(img_path)
                    label_nib = segment_image(
                        nib_img
                    )

                    out_dir = "output/totalseg"
                    os.makedirs(out_dir, exist_ok=True)
                    temp_seg_path = os.path.join(out_dir, f"totalseg_{idx}.nii.gz")
                    nib.save(label_nib, temp_seg_path)

                    io_dict["gt"] = temp_seg_path
                    sample = transforms(io_dict)

                    label_vol = sample["gt"].cpu().squeeze()

                else:
                    raise NotImplementedError(f"mode {mode} not supported")
            ct_vol = sample["image"].cpu().squeeze()

            # Convert back to SimpleITK, restoring origin/spacing/direction
            image_sitk = sitk.GetImageFromArray(ct_vol)

            label_sitk = sitk.GetImageFromArray(label_vol)

            if len(image_sitk.GetSize()) == 3:
                image_sitk.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
                label_sitk.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            else:
                image_sitk.SetDirection((1, 0, 0, 1))
                label_sitk.SetDirection((1, 0, 0, 1))

            if binarize:
                label_sitk = sitk.Cast(label_sitk > 0, sitk.sitkInt64)

            labelmap = sitk.Cast(
                one_hot_labelmap(label_sitk, smoothing_sigma=label_smoothing, num_classes=num_classes),
                sitk.sitkVectorFloat32,
            )
            sample = {'image': image_sitk, 'labelmap': labelmap, 'fname': img_fname}
            self.samples.append(sample)

            gc.collect()
            torch.cuda.empty_cache()

        if not is_labelled:
            del model
        gc.collect()
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]

        image = torch.from_numpy(sitk.GetArrayFromImage(sample['image'])).unsqueeze(0)

        if len(image.size()) == 4:
            labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(3, 0, 1, 2)
        else:
            labelmap = torch.from_numpy(sitk.GetArrayFromImage(sample['labelmap'])).permute(2, 0, 1)

        if self.augmentation:
            device = "cuda" if torch.cuda.is_available() else "cpu" 

            subject_dict = {
                "image": image.to(device),
                "labelmap": labelmap.to(device),
            }

            subject = self.augment(subject_dict)
            def strip_meta(x):
                return x.as_tensor() if isinstance(x, MetaTensor) else x

            # Dynamically handle missing keys
            keys = ['image', 'labelmap']

            subject = {k: strip_meta(subject[k].data).cpu() for k in keys}
            return {'image': subject['image'], 'labelmap': subject['labelmap'], 'fname': sample['fname']}
        else:
            return {'image': image, 'labelmap': labelmap, 'fname': sample['fname']}

    def get_sample(self, item):
        return self.samples[item]


def segment_image(nifti_img):
    segmentation_nifti = totalsegmentator(nifti_img, task='heartchambers_highres', quiet=True)
    return segmentation_nifti