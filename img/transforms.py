from img.processing import resample_image, resample_image_d 
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from monai.transforms import MapTransform

class Resamplerd(MapTransform):
    """
    Resamples each array in `keys` to `out_spacing` and (optionally) `out_size`,
    using nearest‐neighbour if `is_label=True`, else linear interpolation.
    """
    def __init__(self, keys, out_spacing, out_size=None, is_label=False, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.out_spacing = out_spacing
        self.out_size     = out_size
        self.is_label     = is_label

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            # your existing resample_image takes (array, spacing, size, is_label)
            d[key] = resample_image_d(
                d[key],
                self.out_spacing,
                self.out_size,
                self.is_label,
            )
        return d

class Resampler(object):
    """Resamples an image to given element spacing and size."""

    def __init__(self, out_spacing, out_size=None, is_label=False):
        """
        Args:
        :param out_spacing (tuple): Output element spacing.
        :param out_size (tuple, option): Output image size.
        :param is_label (boolean, option): Indicates label maps with nearest neighbor interpolation.
        """
        self.out_spacing = out_spacing
        self.out_size = out_size
        self.is_label = is_label

    def __call__(self, image):
        if isinstance(image, nib.Nifti1Image):
            data = image.get_fdata()
            original_spacing = image.header.get_zooms()[:3]
            zoom_factors = [original_spacing[i] / self.out_spacing[i] for i in range(3)]

            resampled_data = zoom(data, zoom_factors, order=0 if self.is_label else 3)  # Nearest or cubic interpolation

            new_affine = np.copy(image.affine)
            new_affine[:3, :3] = np.diag(self.out_spacing)  # Update spacing in affine

            image_resampled = nib.Nifti1Image(resampled_data, new_affine)
        else:
            image_resampled = resample_image(image, self.out_spacing, self.out_size, self.is_label)

        return image_resampled

class Normalizer(object):
    """Normalizes image intensities with a given function."""

    def __init__(self, transform):
        """
        Args:
        :param transform (callable): Intensity normalization function.
        """
        self.transform = transform

    def __call__(self, image, mask=None):
        image_normalized = self.transform(image, mask)

        return image_normalized

