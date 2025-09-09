import cv2
import numpy as np

def pad_image(img, border, color=0):

    h, w = img.shape
    shape = np.array((h, w))
    ret = color * np.ones(shape + 2 * border)
    ret[border:h+border, border:w+border] = img
    return ret

def correlation(w, f, padding='zero'):
    M, N = f.shape
    m, n = w.shape
    a, b = m // 2, n // 2

    # aplicar padding simétrico
    f_padded = cv2.copyMakeBorder(f, a, a, b, b, cv2.BORDER_CONSTANT)

    g = np.zeros_like(f, dtype=np.float64)
    for y in range(M):
        for x in range(N):
            g[y, x] = (w * f_padded[y:y + m, x:x + n]).sum()

    return g

def convolution(w, f):

    M, N = f.shape       # tamaño de la imagen
    m, n = w.shape       # tamaño del kernel
    a = m // 2           # offset vertical
    b = n // 2           # offset horizontal

    # aplica zero padding
    f_padded = cv2.copyMakeBorder(f, a, a, b, b, cv2.BORDER_CONSTANT)

    g = np.zeros_like(f, dtype=np.float64)

    for y in range(M):
        for x in range(N):
            v = 0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    v += w[s + a, t + b] * f_padded[y - s + a, x - t + b]
            g[y, x] = v

    return g