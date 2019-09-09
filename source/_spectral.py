import numpy as np
from scipy import fftpack
from sklearn.decomposition import PCA


def _rgb2gray(img):
    return np.dot(img.transpose(1, 2, 0)[...,:3], [0.2989, 0.5870, 0.1140])

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
    
def get_spectral_histogram(image):
    image_grey = _rgb2gray(image)
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(image_grey)

    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)

    # Calculate a 2D power spectrum
    psd2D = np.abs(F2)**2

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(psd2D)
    return psd1D
    
def apply_lpf(image):
    im_fft = fftpack.fft2(image)
    # Define the fraction of coefficients (in each direction) we keep
    keep_fraction = 0.1
    # Call ff a copy of the original transform. Numpy arrays have a copy
    # method for this purpose.
    im_fft2 = im_fft.copy()
    # Set r and c to be the number of rows and columns of the array.
    _, r, c = im_fft2.shape
    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[:, int(r*keep_fraction):int(r*(1-keep_fraction)), :] = 0
    # Similarly with the columns:
    im_fft2[:, :, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    return fftpack.ifft2(im_fft2).real
    

def apply_pca(image, n_components=1):
    img_r = np.reshape(image, (image.shape[0], image.shape[1] * image.shape[2]))
    
    pca = PCA(n_components).fit(img_r)
    img_c = pca.transform(img_r)

    img_rec = pca.inverse_transform(img_c)
    img_pca = np.reshape(img_rec, image.shape)    
    return img_pca, np.sum(pca.explained_variance_ratio_)