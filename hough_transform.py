import numpy as np

def hough_transform(img):
    """

    transform a point (x,y) to a sine-like function in Hough space using parameterization:
        rho = x * cos(theta) + y * sin(theta)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache valuess
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    for k in range(len(xs)):
        x = xs[k]
        y = ys[k]
        rho = np.rint(x*cos_t+y*sin_t) + np.floor(len(rhos)/2)
        for i in range(num_thetas):
            v = int(rho[i])
            accumulator[v,i]+=1
            

    return accumulator, rhos, thetas

