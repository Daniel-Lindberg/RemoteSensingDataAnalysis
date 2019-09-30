"""
Homework 1, ASEN
Data analysis Remote Sensing
Data: 09/21/2019
Author: Daniel Lindberg
"""

# Use PCA function in sklearn.decomposition
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pysptools.noise as ns
import rasterio

from scipy.ndimage import filters
from scipy import misc
from osgeo import gdal
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cross_decomposition import CCA


def MNF(img, n_components):
    mnf = ns.MNF()
    mnf.apply(img)
    r = mnf.get_components(n_components)
    return r

def reshape_as_raster(arr):
    """Returns the array in a raster order
    by swapping the axes order from (rows, columns, bands)
    to (bands, rows, columns)
    Parameters
    ----------
    arr : array-like in the image form of (rows, columns, bands)
    """
    # swap the axes order from (rows, columns, bands) to (bands, rows, columns)
    im = np.transpose(arr, [2,0,1])
    return im

def saveMNF(img, input_raster, given_name):
	# Save TIF image to a nre directory of name MNF
	img2 = img.astype(np.uint8)
	img2 = reshape_as_raster(img2)
   	output = "MNF/" + given_name + "_MNF.tif"
	new_dataset = rasterio.open(output, 'w', driver='GTiff',
               height=input_raster.shape[0], width=input_raster.shape[1],
               count=int(3), dtype=input_raster.dtypes[0],
               crs=input_raster.crs, transform=input_raster.transform)
	new_dataset.write(img2)
  	new_dataset.close()

def noiseGenerator(noise_type,image):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row,col,ch= image.shape
    if noise_type == "gauss":       
        mean = 0.0
        var = 0.01
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="n2":
    	# noise multiplied by bottom and top half images,
		# whites stay white blacks black, noise is added to center
    	img = image[...,::-1]/255.0
    	noise =  np.random.normal(loc=0, scale=1, size=img.shape)
    	img2 = img*2
    	n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
    	return n2
    elif noise_type =="n4":
    	# noise multiplied by bottom and top half images,
		# whites stay white blacks black, noise is added to center
		img=image[...,::-1]/255.0
		noise =  np.random.normal(loc=0, scale=1, size=img.shape)
		img2 = img*2
		n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)
		return n4
    else:
        return image

# Load image
img_path = 'florida_amo_2018082_th.jpg'
img = gdal.Open(img_path)
img_arr = img.ReadAsArray()

# Create data matrix
X = np.reshape(img_arr, (3,-1))

# Compute PCA
pca = PCA(n_components=3)
pca.fit(X)
PC = np.reshape(pca.components_,img_arr.shape)

# Plot
"""
plt.figure()
plt.subplot(2,2,1)
plt.imshow(PC[0,:,:],cmap='gray')
plt.title('PC1')
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(PC[1,:,:],cmap='gray')
plt.title('PC2')
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(PC[2,:,:],cmap='gray')
plt.title('PC3')
plt.axis('off')
plt.subplot(2,2,4)
plt.plot(pca.explained_variance_ratio_)
plt.title('Explained Variance Ratio')
"""

total_img = cv2.imread(img_path)
# Plot
plt.figure()
plt.subplot(1,1,1)
plt.axis('off')
plt.plot(pca.explained_variance_ratio_)
print (pca.components_)
plt.xlabel("selected components")
plt.ylabel("Percentage of variance explained")
plt.title('Explained Variance Ratio (Percentage of variance explained by each of the selected components)')
plt.figure()

plt.subplot(2,2,1)
gauss_noisey_img = noiseGenerator("gauss",total_img)
plt.axis('off')
plt.imshow(gauss_noisey_img)
plt.title("Gaussian noisey image")
plt.subplot(2,2,2)
sandp_noisey_img = noiseGenerator("s&p",total_img)
plt.axis('off')
plt.imshow(sandp_noisey_img)
plt.title("S&P noisey image")
plt.subplot(2,2,3)
n2_noisey_img = noiseGenerator("n2",total_img)
plt.axis('off')
plt.imshow(n2_noisey_img)
plt.title("N2 noisey image")
plt.subplot(2,2,4)
n4_noisey_img = noiseGenerator("n4",total_img)
plt.axis('off')
plt.imshow(n4_noisey_img)
plt.title("N4 noisey image")

mnf_gauss = MNF(gauss_noisey_img, 3)
r = rasterio.open(img_path)  
saveMNF(mnf_gauss, r, "Gauss-Noise")
mnf_sandf = MNF(sandp_noisey_img, 3)
saveMNF(mnf_sandf, r, "SANDF-Noise")
mnf_n2 = MNF(n2_noisey_img, 3)
saveMNF(mnf_n2, r, "N2-Noise")
mnf_n4 = MNF(n4_noisey_img, 3)
saveMNF(mnf_n4, r, "N4-Noise")

# Create data matrix
X_gauss = np.reshape(total_img, (3,-1))
plt.figure()
plt.subplot(4,2,1)
pca_gauss = PCA(n_components=3)
pca_gauss.fit(X_gauss)
PC_gauss = np.reshape(pca_gauss.components_,total_img.shape)
img_mnf_gauss = plt.imread("MNF/Gauss-Noise_MNF.tif")
plt.title("MNF-Gauss")
plt.imshow(img_mnf_gauss)
plt.axis('off')
# Compute PCA
plt.subplot(4,2,2)
plt.plot(pca_gauss.explained_variance_ratio_)
plt.title("Var_Ratio_Gauss")

plt.subplot(4,2,3)
pca_sandp = PCA(n_components=3)
pca_sandp.fit(X_gauss)
PC_sandp = np.reshape(pca_sandp.components_,total_img.shape)
plt.axis('off')
img_mnf_gauss = plt.imread("MNF/SANDF-Noise_MNF.tif")
plt.imshow(img_mnf_gauss)
plt.title("MNF-SANDF")
# Compute PCA
plt.subplot(4,2,4)
plt.plot(pca_sandp.explained_variance_ratio_)
plt.title("Var_Ratio_SANDF")

plt.subplot(4,2,5)
pca_n2 = PCA(n_components=3)
pca_n2.fit(X_gauss)
PC_n2 = np.reshape(pca_n2.components_,total_img.shape)
plt.axis('off')
img_mnf_gauss = plt.imread("MNF/N2-Noise_MNF.tif")
plt.imshow(img_mnf_gauss)
plt.title("MNF-N2")
# Compute PCA
plt.subplot(4,2,6)
plt.plot(pca_n2.explained_variance_ratio_)
plt.title("Var_Ratio_N2")

plt.subplot(4,2,7)
pca_n4 = PCA(n_components=3)
pca_n4.fit(X_gauss)
PC_n4 = np.reshape(pca_n4.components_,total_img.shape)
plt.axis('off')
img_mnf_gauss = plt.imread("MNF/N4-Noise_MNF.tif")
plt.imshow(img_mnf_gauss)
plt.title("MNF-N4")
# Compute PCA
plt.subplot(4,2,8)
plt.plot(pca_n4.explained_variance_ratio_)
plt.title("Var_Ratio_N4")

plt.figure()

""" different kernel types
kernel types linear / poly / rbf / sigmoid / cosine / precomputed
gamma  float, default = 1 / n-features

    Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels.

remove_zero_eig : boolean, default=False

    If True, then all components with zero eigenvalues are removed, so that the number of components in the output may be < n_components (and sometimes even zero due to numerical instability). When n_components is None, this parameter is ignored and components with zero eigenvalues are removed regardless.

degree : int, default=3

    Degree for poly kernels. Ignored by other kernels.

"""
plt.subplot(4,2,1)
kpca_lowg = KernelPCA(n_components=3, kernel="rbf", gamma=15)
X_kpca_lowg = kpca_lowg.fit_transform(X_gauss)
plt.imshow(X_kpca_lowg)
plt.axis('off')
plt.title('rbf-lowg')
plt.subplot(4,2,2)
kpca_highg = KernelPCA(n_components=3, kernel="rbf", gamma=100)
X_kpca_highg = kpca_highg.fit_transform(X_gauss)
plt.imshow(X_kpca_highg)
plt.axis('off')
plt.title('rbf-highg')
plt.subplot(4,2,3)
kpca_poly = KernelPCA(n_components=3, kernel="poly")
X_kpca_poly = kpca_poly.fit_transform(X_gauss)
plt.imshow(X_kpca_poly)
plt.axis('off')
plt.title('poly')
plt.subplot(4,2,4)
kpca_gauss_sig = KernelPCA(n_components=3, kernel="sigmoid")
X_kpca_sig = kpca_gauss_sig.fit_transform(X_gauss)
plt.imshow(X_kpca_sig)
plt.axis('off')
plt.title('sigmoid')
plt.subplot(4,2,5)
kpca_gauss_poly_highd = KernelPCA(n_components=3, kernel="poly", degree=10)
X_kpca_poly_highd = kpca_gauss_poly_highd.fit_transform(X_gauss)
plt.imshow(X_kpca_poly_highd)
plt.axis('off')
plt.title('poly-highd')
plt.subplot(4,2,6)
kpca_gauss_poly_lowd = KernelPCA(n_components=3, kernel="poly", degree=4)
X_kpca_poly_lowd = kpca_gauss_poly_lowd.fit_transform(X_gauss)
plt.imshow(X_kpca_poly_lowd)
plt.axis('off')
plt.title('poly-lowd')
print "RBF-LowGamma",X_kpca_lowg
print "RBF-HighGamma",X_kpca_highg
print "Polynomials",X_kpca_poly
print "Sigmoid",X_kpca_sig
print "Poly-HighDegrees",X_kpca_poly_highd
print "Poly-LowDegrees",X_kpca_poly_lowd
print "KPCAs:"
print "RBF-LowGamma:"+str(np.var(X_kpca_lowg))
print "RBF-HighGamma:"+str(np.var(X_kpca_highg))
print "Polynomials:"+str(np.var(X_kpca_poly))
print "Sigmoid:"+str(np.var(X_kpca_sig))
print "Poly-HighDegrees:"+str(np.var(X_kpca_poly_highd))
print "Poly-LowDegrees:"+str(np.var(X_kpca_poly_lowd))

plt.show()