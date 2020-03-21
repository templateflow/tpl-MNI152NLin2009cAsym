from templateflow import api
from scikit.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter
import nibabel as nb
import numpy as np
from skimage.morphology import ball
from scikit.ndimage import binary_erosion
from scipy.ndimage import morphology as m

path = api.get('MNI152NLin2009cAsym', desc='brain', resolution=1, suffix='mask')
im = nb.load(path)
data = np.asanyarray(im.dataobj).astype(bool)
dilate1 = m.binary_dilation(data == 1, ball(1))
dilate2 = m.binary_dilation(dilate1, ball(1))
soft = 0.95 * dilate2
soft[dilate1] = 1.0
gauss = gaussian_filter(soft, 1)
nb.Nifti1Image(gauss, im.affine, im.header).to_filename('gauss.nii.gz')

