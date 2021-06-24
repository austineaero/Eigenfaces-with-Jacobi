"""Generate Eigenfaces from a set of training images."""
from imageio import imread
import numpy as np
from numpy import array,identity,diagonal
from math import sqrt
import random
import util
from sklearn.decomposition import PCA
import os
from PIL import Image
np.random.seed(42)
random.seed(42)

NB_COMPONENTS = 8
SCALE = 32

def main():
    """Load example faces, extract principal components, create Eigenfaces from
    PCs, plot results."""
    # load images, resize them, convert to 1D-vectors
    images_filepaths = get_images_in_directory("images/faces/")
    images = [imread(fp, as_gray=True) for fp in images_filepaths]
    images = [Image.fromarray(image).resize((SCALE, SCALE)) for image in images]
    images_vecs = np.array([image.getdata() for image in images])

    # ------------
    # Custom Implementation of PCA
    # ------------
    pcs, images_vecs_transformed, images_vecs_reversed = custom_pca(images_vecs, NB_COMPONENTS)
    pcs = pcs.reshape((NB_COMPONENTS, SCALE, SCALE))

    # plot (First example image, recovered first example image, first 8 PCs)
    plots_imgs = [images[0], images_vecs_reversed[0].reshape((SCALE, SCALE))]
    plots_titles = ["Image 0", "Image 0\n(reconstructed)"]
    for i in range(NB_COMPONENTS):
       plots_imgs.append(pcs[i])
       plots_titles.append("Eigenface %d" % (i))

    util.plot_images_grayscale(plots_imgs, plots_titles)


    # ------------
    # Using the PCA implementation from scikit
    # ------------
    # train PCA, embed image vectors, reverse the embedding (lossy)
    pca = PCA(NB_COMPONENTS)
    images_vecs_transformed = pca.fit_transform(images_vecs)
    images_vecs_reversed = pca.inverse_transform(images_vecs_transformed)

    # Extract Eigenfaces. The Eigenfaces are the principal components.
    pcs = pca.components_.reshape((NB_COMPONENTS, SCALE, SCALE))

    # plot (First example image, recovered first example image, first 8 PCs)
    plots_imgs = [images[0], images_vecs_reversed[0].reshape((SCALE, SCALE))]
    plots_titles = ["Image 0", "Image 0\n(reconstructed)"]
    for i in range(NB_COMPONENTS):
       plots_imgs.append(pcs[i])
       plots_titles.append("Eigenface %d" % (i))

    util.plot_images_grayscale(plots_imgs, plots_titles)


def jacobi(a,tol = 1.0e-9): # Jacobi method

    def maxElem(a): # Find largest off-diag. element a[k,l]
        n = len(a)
        aMax = 0.0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(a[i,j]) >= aMax:
                    aMax = abs(a[i,j])
                    k = i; l = j
        return aMax,k,l

    def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
        n = len(a)
        aDiff = a[l,l] - a[k,k]
        if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
        else:
            phi = aDiff/(2.0*a[k,l])
            t = 1.0/(abs(phi) + sqrt(phi**2 + 1.0))
            if phi < 0.0: t = -t
        c = 1.0/sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = a[k,l]
        a[k,l] = 0.0
        a[k,k] = a[k,k] - t*temp
        a[l,l] = a[l,l] + t*temp
        for i in range(k):      # Case of i < k
            temp = a[i,k]
            a[i,k] = temp - s*(a[i,l] + tau*temp)
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(k+1,l):  # Case of k < i < l
            temp = a[k,i]
            a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(l+1,n):  # Case of i > l
            temp = a[k,i]
            a[k,i] = temp - s*(a[l,i] + tau*temp)
            a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
        for i in range(n):      # Update transformation matrix
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
        
    n = len(a)
    maxRot = 5*(n**2)       # Set limit on number of rotations
    p = identity(n)*1.0     # Initialize transformation matrix
    for i in range(maxRot): # Jacobi rotation loop 
        aMax,k,l = maxElem(a)
        if aMax < tol: return diagonal(a),p
        rotate(a,p,k,l)
    print('Jacobi method did not converge')
       

def custom_pca(images_vecs, nb_components):
    """Custom implementation of PCA for images.
    Args:
       images_vecs    The images to transform.
       nb_components  Number of principal components.
    Returns:
       Principal Components of shape (NB_COMPONENTS, height*width),
       Transformed images of shape (nb_images, NB_COMPONENTS),
       Reverse transformed/reconstructed images of shape (nb_images, height*width)
    """
    imgs = np.copy(images_vecs)
    imgs = np.transpose(imgs)
    mean = np.average(imgs, axis=1)
    mean = mean[:, np.newaxis]
    A = imgs - np.tile(mean, (1, imgs.shape[1]))
    print(np.shape(A))
    
    # Compute eigenvectors of A^TA instead of AA^T (covariance matrix) as
    # that is faster.
    #B = A[:30,:30]
    L = np.dot(np.transpose(A), A) # A^TA (10x10 matrix)
    #eigenvalues, eigenvectors = np.linalg.eig(L)
    eigenvalues, eigenvectors = jacobi(L)
    print(eigenvalues)
    print(eigenvectors)

    # recover eigenvectors of AA^T (covariance matrix)
    U = np.dot(A, eigenvectors)

    # reduce to requested number of eigenvectors
    U = np.transpose(U)
    nb_components = min(len(eigenvectors), nb_components)
    U = U[0:nb_components, :]

    # project faces to face space
    imgs_transformed = np.dot(U, A)
    imgs_transformed = np.transpose(imgs_transformed)

    # reconstruct faces
    imgs_reversed = np.dot(np.transpose(U), np.transpose(imgs_transformed))
    imgs_reversed = np.transpose(imgs_reversed)

    return U, imgs_transformed, imgs_reversed

def get_images_in_directory(directory, ext="jpg"):
    """Read the filepaths to all images in a directory.
    Args:
       directory Filepath to the directory.
       ext File extension of the images.
    Returns:
       List of filepaths
    """
    filepaths = []
    for fname in os.listdir(directory):
       if fname.endswith(ext) and os.path.isfile(os.path.join(directory, fname)):
          filepaths.append(os.path.join(directory, fname))
    return filepaths

if __name__ == "__main__":
    main()