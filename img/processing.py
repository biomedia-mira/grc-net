import numpy as np
import SimpleITK as sitk
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as _cp_gaussian

import torch.nn.functional as F

from monai.transforms import (
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    ThresholdIntensityd,
)
from monai.data import MetaTensor

def resample_image_d(
    image,
    out_spacing=(1.0, 1.0, 1.0),
    out_size=None,
    is_label=False,
    pad_value=0,
):
    """
    Resample a MetaTensor to given spacing and (optionally) size, in a fashion similar to SITK.
    """

    orig_affine = np.array(image.meta.get("affine"))
    orig_spacing = np.linalg.norm(orig_affine[:3, :3], axis=0)
    direction = orig_affine[:3, :3] / orig_spacing
    origin = orig_affine[:3, 3]

    orig_size = np.array(image.shape[1:], dtype=int)

    if out_size is None:
        out_size = np.round(orig_size * orig_spacing / np.array(out_spacing)).astype(int)
    else:
        out_size = np.array(out_size, dtype=int)

    orig_center = (orig_size - 1) / 2.0 * orig_spacing
    out_center  = (out_size  - 1) / 2.0 * np.array(out_spacing)

    orig_ctr_phys = direction.dot(orig_center)
    out_ctr_phys  = direction.dot(out_center)
    new_origin = origin + (orig_ctr_phys - out_ctr_phys)

    new_affine = np.eye(4, dtype=float)
    new_affine[:3, :3] = direction * np.array(out_spacing)
    new_affine[:3, 3]  = new_origin

    img = image.clone().detach()
    dtype = img.dtype
    if not is_label:
        img = img.float()
    img = img.unsqueeze(0)

    mode = "nearest" if is_label else "trilinear"
    align = False if mode == "trilinear" else None
    resized = F.interpolate(img, size=tuple(out_size.tolist()), mode=mode, align_corners=align)

    resized = resized.squeeze(0)
    if is_label:
        resized = resized.long()

    new_meta = dict(image.meta)
    new_meta["affine"] = new_affine
    new_meta["original_affine"] = new_affine
    new_meta["spatial_shape"] = tuple(out_size.tolist())

    return MetaTensor(resized, meta=new_meta)


zero_mean_unit_var = NormalizeIntensityd(
    keys=["image"],
    nonzero=True,        
    channel_wise=False,  
)

range_matching = ScaleIntensityRangePercentilesd(
    keys=["image"],
    lower=4.0,    
    upper=96.0,   
    b_min=0.0,    
    b_max=1.0,    
    clip=True,    
)

zero_one = ScaleIntensityRanged(
    keys=["image"],
    a_min=None,
    a_max=None,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

threshold_zero = ThresholdIntensityd(
    keys=["image"],
    threshold=0.0,
    above=False,
    cval=0.0,
)

def same_image_domain(image1, image2):
    """Checks whether two images cover the same physical domain."""

    same_size = image1.GetSize() == image2.GetSize()
    same_spacing = image1.GetSpacing() == image2.GetSpacing()
    same_origin = image1.GetOrigin() == image2.GetOrigin()
    same_direction = image1.GetDirection() == image2.GetDirection()

    return same_size and same_spacing and same_origin and same_direction


def reorient_image(image):
    """Reorients an image to standard radiology view."""

    dir = np.array(image.GetDirection()).reshape(len(image.GetSize()), -1)
    ind = np.argmax(np.abs(dir), axis=0)
    new_size = np.array(image.GetSize())[ind]
    new_spacing = np.array(image.GetSpacing())[ind]
    new_extent = new_size * new_spacing
    new_dir = dir[:, ind]

    flip = np.diag(new_dir) < 0
    flip_diag = flip * -1
    flip_diag[flip_diag == 0] = 1
    flip_mat = np.diag(flip_diag)

    new_origin = np.array(image.GetOrigin()) + np.matmul(new_dir, (new_extent * flip))
    new_dir = np.matmul(new_dir, flip_mat)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(new_dir.flatten().tolist())
    resample.SetOutputOrigin(new_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(image)


def resample_image_to_ref(image, ref, is_label=False, pad_value=0):
    """Resamples an image to match the resolution and size of a given reference image."""

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)

def resample_image(image, out_spacing=(1.0, 1.0, 1.0), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        return resample.Execute(image)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        if image.GetNumberOfComponentsPerPixel() == 1:
            return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
        else:
            return resample.Execute(sitk.Cast(image, sitk.sitkVectorFloat32))


def extract_patch(image, pixel, out_spacing=(1.0, 1.0, 1.0), out_size=(32, 32, 32), is_label=False, pad_value=0):
    """Extracts a patch of given resolution and size at a specific location."""

    original_spacing = np.array(image.GetSpacing())

    out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    pos = np.matmul(original_direction, np.array(pixel) * np.array(original_spacing)) + np.array(image.GetOrigin())
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(pos - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    if image.GetNumberOfComponentsPerPixel() == 1:
        return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
    else:
        return resample.Execute(sitk.Cast(image, sitk.sitkVectorFloat32))

def one_hot_labelmap(labelmap, smoothing_sigma=0.0, num_classes=33, dtype=cp.float32):
    """
    GPU‑accelerated one‑hot encode + optional Gaussian smoothing.
    """

    lab_cpu = sitk.GetArrayFromImage(labelmap)
    lab = cp.asarray(lab_cpu)
    labels = cp.unique(lab)

    if len(labels) != num_classes:
        raise ValueError(f"Expected {num_classes} classes but found {len(labels)} in labelmap.")

    # build one‐hot on GPU
    one_hot = (lab[..., None] == labels).astype(dtype)

    if smoothing_sigma > 0:
        # don't blur across the channel axis
        sigmas = (smoothing_sigma,)*3 + (0.0,)
        one_hot = _cp_gaussian(one_hot, sigma=sigmas, mode='nearest')

    one_hot_cpu = cp.asnumpy(one_hot)
    out = sitk.GetImageFromArray(one_hot_cpu, isVector=True)
    out.CopyInformation(labelmap)
    return out
