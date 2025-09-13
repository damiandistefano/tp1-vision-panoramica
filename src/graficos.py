import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .utils import kp_to_xy as _kp_xy


def graficar_puntos(imgs, pts, title="Puntos seleccionados", point_radius=5, figsize=(12,6)):
    """
    Dibuja puntos (x,y) sobre varias imágenes.
    - imgs: lista de np.ndarray BGR o RGB
    - pts: lista de arrays (N_i,2) con coordenadas (x,y) para cada imagen
    - title: título superior
    - point_radius: radio de los puntos
    - figsize: tamaño figura matplotlib
    """
 
    n_imgs = len(imgs)
    fig, axes = plt.subplots(1, n_imgs, figsize=figsize)
    if n_imgs == 1:
        axes = [axes]

    for ax, img, pts_img in zip(axes, imgs, pts):
        # mostrar la imagen (cuidado si es BGR)
        if img.shape[2] == 3:
            ax.imshow(img[..., ::-1])  # de BGR a RGB
        else:
            ax.imshow(img, cmap='gray')
        ax.axis('off')


        
        pts_img = np.array([k.pt for k in pts_img], dtype=float)

        # dibujar puntos y numerarlos
        for i, (x, y) in enumerate(pts_img):
            ax.add_patch(mpatches.Circle((x, y), radius=point_radius, color='red'))

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()



def match_descriptors(desc1,desc2,kps1,kps2,
    ratio_thresh = 0.75,
    cross_check= True):
    """
    Matching entre desc1 (imagen A) y desc2 (imagen B)

    Args:
        desc1, desc2: descriptores. 
                    
        kps1, kps2: listas de cv2.KeyPoint (o None si no se quieren coordenadas).
        ratio_thresh: umbral de Lowe 
        cross_check: si True realiza verificación mutua .

    Returns:
        matches_good: lista de cv2.DMatch (ordenada por distancia ascendente).
        pts1: Nx2 array con coordenadas en imagen 1 (o None si no se pasaron kps).
        pts2: Nx2 array con coordenadas en imagen 2 (o None si no se pasaron kps).
    """

    # Detectar tipo de descriptores -> norma para BFMatcher
    norm = cv2.NORM_L2

    matcher = cv2.BFMatcher(norm, crossCheck=False)

    knn1 = matcher.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    candidates = []
    for m_n in knn1:
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < ratio_thresh * n.distance:
            candidates.append(m)

    if not cross_check:
        matches_good = sorted(candidates, key=lambda x: x.distance)
    else:
        # Reverse best: desc2 -> desc1 (k=1) para comprobar mutual best
        knn2 = matcher.knnMatch(desc2, desc1, k=1)
        reverse_best = {}
        for ll in knn2:
            if len(ll) == 0:
                continue
            rm = ll[0]
            # rm.queryIdx: idx en desc2; rm.trainIdx: idx en desc1
            reverse_best[rm.queryIdx] = rm.trainIdx

        # Filtrar sólo matches mutuos
        mutual = []
        for m in candidates:
            # m.queryIdx: idx en desc1; m.trainIdx: idx en desc2
            rev = reverse_best.get(m.trainIdx, None)
            if rev is not None and rev == m.queryIdx:
                mutual.append(m)
        matches_good = sorted(mutual, key=lambda x: x.distance)


    # Si se pasaron keypoints, devolver además coordenadas (x,y)
    if kps1 is not None and kps2 is not None:
        pts1 = np.array([kps1[m.queryIdx].pt for m in matches_good], dtype=np.float32)
        pts2 = np.array([kps2[m.trainIdx].pt for m in matches_good], dtype=np.float32)
    else:
        pts1 = pts2 = None

    return matches_good, pts1, pts2



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

    # Dibujar puntos y líneas
    for i, ((xA, yA), (xB, yB)) in enumerate(zip(pts_A, pts_B)):
        col = colors[i % len(colors)]
        # A
        ax.add_patch(mpatches.Circle((xA, yA), radius=point_radius, color=col))
        # B (con offset horizontal wA)
        ax.add_patch(mpatches.Circle((wA + xB, yB), radius=point_radius, color=col))
        # línea
        ax.plot([xA, wA + xB], [yA, yB], '-', color=col, linewidth=2)

    plt.tight_layout()

  
    fig.canvas.draw()
    img_out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_out = img_out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_out



def draw_matches_ransac(imgA, kpA, imgB, kpB, matches_or_pairs, ransac_result,
                        title=None, kp_radius=4, line_thickness=1):
    """
    Dibuja collage A|B con inliers (verde) y outliers (rojo) según RANSAC.
    
    - matches_or_pairs: lista de cv2.DMatch o array (M,2) de índices
    - ransac_result: dict devuelto por ransac_homography
    """
    # --- construir pairs_idx ---
    if isinstance(matches_or_pairs, (list,tuple)) and len(matches_or_pairs)>0 and hasattr(matches_or_pairs[0],"queryIdx"):
        pairs_idx = np.array([[int(m.queryIdx), int(m.trainIdx)] for m in matches_or_pairs], dtype=np.int32)
    else:
        pairs_idx = np.asarray(matches_or_pairs, dtype=np.int32)
    M = pairs_idx.shape[0]

    # --- máscara de inliers ---
    mask = ransac_result.get('mask', None)
    if mask is None or len(np.asarray(mask).reshape(-1)) != M:
        mask = np.zeros((M,), dtype=bool)
        idxs = ransac_result.get('inliers_idx', None)
        if idxs is not None:
            mask[np.asarray(idxs, dtype=int)] = True
    else:
        mask = np.asarray(mask).reshape(-1).astype(bool)

  
    # --- armar canvas ---
    hA,wA = imgA.shape[:2]; hB,wB = imgB.shape[:2]
    Hh, Ww = max(hA,hB), wA+wB
    canvas = np.zeros((Hh,Ww,3), dtype=np.uint8)
    canvas[:hA,:wA] = imgA
    canvas[:hB,wA:wA+wB] = imgB

    # --- coords de keypoints ---
    xyA = np.array([kp.pt for kp in kpA], dtype=np.float32)
    xyB = np.array([kp.pt for kp in kpB], dtype=np.float32)

    # --- dibujar matches ---
    for (iA,iB), ok in zip(pairs_idx, mask):
        x1,y1 = xyA[iA]; x2,y2 = xyB[iB]
        color = (0,255,0) if ok else (0,0,255)  # verde= inlier, rojo= outlier
        cv2.line(canvas, (int(x1),int(y1)), (int(wA+x2),int(y2)),
                 color, thickness=line_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(x1),int(y1)), kp_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(wA+x2),int(y2)), kp_radius, color, -1, cv2.LINE_AA)

    if title:
        cv2.putText(canvas, title, (16,36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    return canvas




def stitch_three_images(imgL, imgC, imgR, H_LC, H_RC):
    """
    Une 3 imágenes en una panorámica, usando homografías al plano de la imagen central.

    Args:
        imgL, imgC, imgR: imágenes izquierda, centro y derecha (np.uint8, BGR o gray).
        H_LC: homografía que mapea puntos de imgL -> imgC
        H_RC: homografía que mapea puntos de imgR -> imgC

    Returns:
        panorama: imagen panorámica (np.uint8, BGR).
    """
    # asegurar 3 canales
    def _bgr(img):
        if img.ndim==2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2]==4: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    imgL, imgC, imgR = _bgr(imgL), _bgr(imgC), _bgr(imgR)

    hC, wC = imgC.shape[:2]

    # esquinas de cada imagen
    def corners(img):
        h, w = img.shape[:2]
        return np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)

    cornersC = corners(imgC)
    cornersL_warp = cv2.perspectiveTransform(corners(imgL), H_LC)
    cornersR_warp = cv2.perspectiveTransform(corners(imgR), H_RC)

    all_corners = np.vstack((cornersC, cornersL_warp, cornersR_warp))

    # calcular bounding box global
    [xmin, ymin] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    # compensar traslación para evitar índices negativos
    tx, ty = -xmin if xmin<0 else 0, -ymin if ymin<0 else 0
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)

    width, height = xmax - xmin, ymax - ymin

    # warpear L y R
    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    warpL = cv2.warpPerspective(imgL, T @ H_LC, (width, height))
    warpR = cv2.warpPerspective(imgR, T @ H_RC, (width, height))
    warpC = cv2.warpPerspective(imgC, T, (width, height))  # solo traslación

    # combinar (superposición simple: donde hay píxeles ≠0, se quedan)
    maskL = (warpL>0).astype(np.uint8)
    maskR = (warpR>0).astype(np.uint8)
    maskC = (warpC>0).astype(np.uint8)

    panorama = np.where(maskC, warpC, panorama)
    panorama = np.where(maskL, warpL, panorama)
    panorama = np.where(maskR, warpR, panorama)

    return panorama
