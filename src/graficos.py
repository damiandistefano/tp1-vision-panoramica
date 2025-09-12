import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .utils import kp_to_xy as _kp_xy



def matching(img1, kp1, des1, img2, kp2, des2):
    """
    Realiza matching entre dos imágenes usando:
    1. Fuerza bruta + cross-check
    2. Lowe ratio test para filtrar matches

    Retorna:
        res_img_manual: imagen con matches por fuerza bruta + cross-check
        res_img_ratio: imagen con matches filtrados por Lowe ratio test
    """

    # =========================
    # Matching manual (Fuerza bruta + cross-check)
    # =========================
    candidate_matches = []

    for i, d_obj in enumerate(des1):
        distances = np.linalg.norm(des2 - d_obj, axis=1)
        min_index = np.argmin(distances)
        match = cv2.DMatch(_queryIdx=i, _trainIdx=min_index, _distance=distances[min_index])
        candidate_matches.append(match)

    matches = []
    for match in candidate_matches:
        i = match.queryIdx
        min_index = match.trainIdx
        reverse_distances = np.linalg.norm(des1 - des2[min_index], axis=1)
        reverse_min_index = np.argmin(reverse_distances)
        if reverse_min_index == i:
            matches.append(match)

    matches = sorted(matches, key=lambda x: x.distance)

    match_color = (0, 0, 255)  # rojo
    keypoint_color = (0, 255, 0)  # verde

    res_img_manual = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        matchColor=match_color,
        singlePointColor=keypoint_color,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # =========================
    # Lowe ratio test usando BFMatcher
    # =========================
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for _ in range(len(knn_matches))]
    for i, (m, n) in enumerate(knn_matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),       # líneas verdes
                       singlePointColor=(255, 0, 0), # keypoints rojos
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    res_img_ratio = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2, knn_matches, None, **draw_params
    )

    return res_img_manual, res_img_ratio


def draw_matches_side_by_side(
    imgA, imgB,
    pts_A, pts_B,
    title="Correspondencias manuales",
    figsize=(12,6),
    point_radius=6,
):
    """
    Dibuja un collage A|B y une pts_A[i] con pts_B[i] (mismo orden).
    - imgA_path, imgB_path: rutas a las imágenes.
    - pts_A, pts_B: arrays shape (N,2), coordenadas (x,y) en cada imagen.
    - out_path: si se da, guarda el collage en disco.
    - title: título superior.
    - return_image: si True, devuelve el collage como np.ndarray (RGB).
    """


    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]

    # Canvas lado a lado
    H = max(hA, hB)
    W = wA + wB
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:hA, :wA] = imgA
    canvas[:hB, wA:wA+wB] = imgB

    # Paleta de colores cíclica
    colors = np.array([
        (255,   0,   0),   # rojo
        (  0, 255,   0),   # verde
        (  0, 128, 255),   # naranja/azul claro
        (255,   0, 255),   # magenta
        (255, 255,   0),   # amarillo
        (  0, 255, 255),   # cyan
    ], dtype=np.float32) / 255.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas)
    ax.axis('off')
    ax.set_title(title, fontsize=14)

    pts_A = np.asarray(pts_A, dtype=float)
    pts_B = np.asarray(pts_B, dtype=float)
    assert pts_A.shape == pts_B.shape and pts_A.ndim == 2 and pts_A.shape[1] == 2, "pts_A/pts_B deben ser (N,2)"

    # Dibujar puntos y líneas
    for i, ((xA, yA), (xB, yB)) in enumerate(zip(pts_A, pts_B)):
        col = colors[i % len(colors)]
        # A
        ax.add_patch(mpatches.Circle((xA, yA), radius=point_radius, color=col))
        ax.text(xA + 8, yA - 8, str(i+1), color=col, fontsize=12, weight='bold')
        # B (con offset horizontal wA)
        ax.add_patch(mpatches.Circle((wA + xB, yB), radius=point_radius, color=col))
        ax.text(wA + xB + 8, yB - 8, str(i+1), color=col, fontsize=12, weight='bold')
        # línea
        ax.plot([xA, wA + xB], [yA, yB], '-', color=col, linewidth=2)

    plt.tight_layout()

  
    fig.canvas.draw()
    img_out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_out = img_out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_out


def draw_inliers_one_figure(imgA, kpA, imgB, kpB, pairs_idx, inliers_mask,
                            title=None, kp_radius=5, line_thickness=2):
    """
    Dibuja collage A|B con SOLO inliers (verde).
    - pairs_idx: (M,2) índices (iA,iB) de good matches usados en RANSAC
    - inliers_mask: (M,) bool indicando qué pares son inliers
    """
    A, B = imgA.copy(), imgB.copy()
    hA,wA = A.shape[:2]; hB,wB = B.shape[:2]
    Hh, Ww = max(hA,hB), wA+wB
    canvas = np.zeros((Hh,Ww,3), dtype=np.uint8)
    canvas[:hA,:wA] = A
    canvas[:hB,wA:wA+wB] = B

    xyA, xyB = _kp_xy(kpA), _kp_xy(kpB)
    inliers_mask = np.asarray(inliers_mask, dtype=bool)

    for (iA,iB), ok in zip(pairs_idx, inliers_mask):
        if not ok: 
            continue
        x1,y1 = xyA[iA]; x2,y2 = xyB[iB]
        # línea y endpoints en verde
        cv2.line(canvas, (int(x1),int(y1)), (int(wA+x2),int(y2)),
                 (0,255,0), thickness=line_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(x1),int(y1)), kp_radius, (0,255,0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(wA+x2),int(y2)), kp_radius, (0,255,0), -1, cv2.LINE_AA)

    if title:
        cv2.putText(canvas, title, (16,36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return canvas
