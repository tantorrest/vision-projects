import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel using zero-padding.

    Returns output of image convolved with kernel
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi), dtype=image.dtype)

    #zero-padding
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    pad_width = int(np.floor(Wk/2))
    pad_height = int(np.floor(Hk/2))
    pad_image=padded
    kernelFlipped = np.flip(np.flip(kernel,0),1)
    evenHeight = 0
    evenWidth = 0
    if Hk%2 == 0:
        evenHeight = 1
    if Wk%2 == 0:
        evenWidth = 1
    for m in range(pad_height, Hi+pad_height):
        for n in range(pad_width, Wi + pad_width):
            image_section = pad_image[m-pad_height+evenHeight:m+pad_height+1, n-pad_width+evenWidth:n+pad_width+1]
            
            imageKernelProduct = np.multiply(image_section, kernelFlipped)
            out[m-pad_height,n-pad_width] = np.sum(imageKernelProduct)
    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel with sigma.

        Returns kernel
    """  
    
    kernel = np.zeros((size, size))
    denominator = 2 * np.pi * sigma**2
    k = (size - 1 ) / 2
    j_ray = np.array(range(size))
    for i in range(size):
        kernel[i, j_ray[:]] = np.exp(((i-k)**2 + (j_ray[:] - k)**2)/(-2*sigma**2)) / denominator

    return kernel

def partial_x(img):
    """ Returns partial x-derivative of input img.
    """

    out = np.zeros(np.size(img), dtype=img.dtype)
    Dx = np.array([[1,0,-1]])
    out = conv(img,Dx)/2

    return out

def partial_y(img):
    """ Returns partial y-derivative of input img.
    """

    out = np.zeros(np.size(img))
    Dy = np.array([[1],[0], [-1]])
    out = conv(img,Dy)/2

    return out

def gradient(img):
    """ Returns gradient magnitude (G) and direction (theta) of input img.

    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy,Gx)
    theta = np.degrees(theta)
    theta[theta < 0] +=360

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression
    Returns non-maximum suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    directionMap = {0:(0,1),45:(1,1),90:(1,0),135:(1,-1),180:(0,-1),225:(-1,-1),270:(-1,0),315:(-1,1),360:(0,1)}
    
    for i in range(H):
        for j in range(W):
            edge_strength = G[i,j]
            direction = theta[i,j]
            dirI,dirJ = directionMap[direction]
            px1 = 0
            px2 = 0
            if not (i+dirI >= H or j+dirJ >= W or i+dirI < 0 or j+dirJ < 0):
                px1 = G[i+dirI, j+dirJ]
            if not (i-dirI < 0 or j-dirJ < 0 or i-dirI >= H or j-dirJ >= W):
                px2 = G[i-dirI, j-dirJ]
            if edge_strength > px1 and edge_strength > px2:
                out[i,j] = edge_strength
            else:
                out[i,j] = 0

    return out

def double_thresholding(img, high, low):
    """
    Helper function for implementing double_thresholding
    Returns:
        strong_edges: Boolean array representing strong edges.
            (pixels with the values above the higher threshold)
        weak_edges: Boolean array representing weak edges.
            (pixels with the values below the higher threshould and 
                above the lower threshold)
    """

    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)
    Hi = img.shape[0]
    Wi = img.shape[1]
    
    for i in range(Hi):
        for j in range(Wi):
            edge_strength = img[i][j]
            if (edge_strength > high):
                strong_edges[i][j] = True
            elif (edge_strength > low):
                weak_edges[i][j] = True

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Method to link strong edges with connected weak edges.

    Returns edges array where pixels containing 1 are edges
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.copy(strong_edges)
    for i,j in indices:
        visited, queue = set(), [(i,j)]
        while queue:
            px = queue.pop(0)
            if px not in visited:
                visited.add(px)
                neighbors = get_neighbors(px[0],px[1], H, W)
                for k,l in neighbors:
                    if weak_edges[k][l] == True:
                        if (k,l) not in visited:
                            edges[k][l]=1
                            queue.extend([(k,l)])

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implements a canny edge detector by using the functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns array of edges, where pixels containing 1 are edges
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nm_suppressed_img = non_maximum_suppression(G,theta)
    strong_edges, weak_edges = double_thresholding(nm_suppressed_img, high, low)
    edges = link_edges(strong_edges, weak_edges)
    return edges


# Load sample image
img = io.imread('iguana.png', as_grey=True)

# Run Canny edge detector
edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(edges)
plt.title('Edges of image')
plt.axis('off')

plt.show()