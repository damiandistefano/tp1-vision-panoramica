import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize(img, w):
    w_, h_ = img.shape[1], img.shape[0]
    s = w / w_                  # factor de escala
    h = int(h_ * s)
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return resized, s


def detectar_caracteristicas(img_color,sift):
    """
    Detecta características SIFT en una imagen.

    Parámetros:
      img_color: imagen en color BGR
      ancho: opcional, ancho al que se quiere redimensionar manteniendo proporción

    Devuelve:
      img_kp: imagen con keypoints dibujados
      kp: lista de keypoints de OpenCV
      responses: array con la respuesta (response) de cada keypoint
    """
    
    # Convertir a gris
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Detectar keypoints y descriptores
    kp, des = sift.detectAndCompute(img_gray, None)

    # Extraer las respuestas
    responses = np.array([k.response for k in kp], dtype=np.float32)

    for k in kp:
        k.size = 30

    # Dibujar los keypoints
    img_kp = cv2.drawKeypoints(
        img_color, kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )

    return img_kp, kp, responses , des



def anms(img,keypoints, responses, N):
    """
    Adaptive Non-Maximal Suppression (ANMS) compatible con:
      - keypoints: lista de cv2.KeyPoint
      - responses: array-like de floats (misma longitud que keypoints)
      - N: número de keypoints que queremos conservar

    Devuelve:
      - keypoints_sel: lista de cv2.KeyPoint seleccionados (ordenados por importancia)
      - responses_sel: np.array con las responses correspondientes
      - indices: np.array de índices seleccionados (en el orden devuelto)
    """
    K = len(keypoints)
    

    responses = np.asarray(responses, dtype=np.float32)
    
    if responses.shape[0] != K:
        raise ValueError("La longitud de 'responses' debe coincidir con la de 'keypoints'")

    # Extraer coordenadas (x,y)
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # shape (K,2)

    # Inicializar radios con infinito
    radii = np.full(K, np.inf, dtype=np.float32)

    # Calcular radio mínimo a un punto más fuerte (j tal que responses[j] > responses[i])
    for i in range(K):
        mask_stronger = (responses) > responses[i]
        # evitar comparar consigo mismo
        mask_stronger[i] = False
        if not np.any(mask_stronger):
            # si no hay puntos "más fuertes", dejamos radii[i] = inf (muy alto)
            radii[i] = np.inf
            continue
        stronger_idx = np.where(mask_stronger)[0]
        # distancias desde i a los más fuertes
        dif = pts[stronger_idx] - pts[i]            # (M,2)
        dists = np.sqrt(np.sum(dif * dif, axis=1)) # (M,)
        radii[i] = np.min(dists)

    # Ordenar por radio descendente (mayor radio = más aislado/importante)
    order = np.argsort(-radii)  # índices de mayor a menor radio
    selected = order[:min(N, K)]

    keypoints_sel = [keypoints[i] for i in selected]
    responses_sel = responses[selected]


    img_anms = cv2.drawKeypoints(
        img,
        keypoints_sel,  
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )


    return img_anms,keypoints_sel, responses_sel, selected




def pick_points_cv(img, win_name="Pick 4 points", n_points=4, out_npy=None):
   
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, str(len(points)), (x+6, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow(win_name, img)

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, img)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        key = cv2.waitKey(20) & 0xFF
        # cerrar al completar o con ESC
        if (len(points) >= n_points) or key == 27:
            break

    cv2.destroyWindow(win_name)
    pts = np.array(points, dtype=np.float64)
    if out_npy:
        np.save(out_npy, pts)
    return pts


# ---------- DLT (sin normalización) ----------
def _homography_dlt(xy1, xy2):
    """
    Estima H (3x3) con DLT (sin normalización).
    xy1 -> xy2  (N>=4)
    """
    xy1 = np.asarray(xy1, dtype=np.float64)
    xy2 = np.asarray(xy2, dtype=np.float64)
    assert xy1.shape == xy2.shape and xy1.shape[0] >= 4

    N = xy1.shape[0]
    A = []
    for i in range(N):
        x, y   = xy1[i]
        u, v   = xy2[i]
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        A.append([x, y, 1,  0,  0,  0, -u*x, -u*y, -u])
    A = np.asarray(A, dtype=np.float64)

    # Resolver Ah=0 con SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)

    # Normalizar tal que H[2,2] = 1 (si se puede)
    if abs(H[2,2]) > 1e-12:
        H = H / H[2,2]
    return H

def reprojection_rmse(H, src, dst):
    N = src.shape[0]
    src_h = np.hstack([src, np.ones((N,1))])
    proj  = (H @ src_h.T).T
    proj  = proj[:, :2] / proj[:, 2:3]
    d = np.linalg.norm(proj - dst, axis=1)  # error por punto en píxeles
    return float(np.sqrt(np.mean(d**2))), d


# ---------- RANSAC ----------
def ransac_homography(pts1, pts2, thresh=3.0, max_iters=2000, confidence=0.999, random_state=None):
    """
    Estima H con RANSAC sin OpenCV.
    pts1, pts2: (N,2) correspondencias
    thresh: umbral de inliers en píxeles (error simétrico <= thresh^2*2 aprox)
    max_iters: tope duro
    confidence: prob. de tener al menos una muestra libre de outliers (ajusta iteraciones adaptativamente)
    return: H_best (3x3), inlier_mask (N,bool), stats (dict)
    """
    rng = np.random.default_rng(random_state)
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    N = pts1.shape[0]
    if N < 4:
        raise ValueError("Se requieren al menos 4 correspondencias.")

    best_H = None
    best_inliers = None
    best_inlier_count = 0

    s = 4  # tamaño de muestra mínima para homografía
    # Iteraciones adaptativas (se actualiza cuando mejora w)
    it = 0
    max_adaptive = max_iters

    while it < max_adaptive:
        it += 1
        # 1) Muestreo mínimo sin reemplazo
        idx = rng.choice(N, size=s, replace=False)
        try:
            H = _homography_dlt(pts1[idx], pts2[idx])
        except np.linalg.LinAlgError:
            continue  # muestra degenerada (colineal, etc.)

        # 2) Medir errores para todos
        errs = _symmetric_transfer_errors(H, pts1, pts2)

        # 3) Inliers
        # Comparación con thresh en píxeles: usamos err <= 2*thresh^2 (dos proyecciones)
        thr2 = 2.0 * (thresh**2)
        inliers = errs <= thr2
        count = int(np.sum(inliers))

        # 4) Actualizar mejor
        if count > best_inlier_count:
            best_inlier_count = count
            best_inliers = inliers
            best_H = H

            # 5) Actualizar nº de iteraciones necesarias (adaptativo)
            w = count / float(N)
            w = np.clip(w, 1e-6, 1-1e-6)
            num = np.log(1 - confidence)
            den = np.log(1 - (w**s))
            max_adaptive = min(max_iters, int(np.ceil(num / den)))
            if max_adaptive <= it:  # ya alcanzamos la confianza deseada
                break

    if best_H is None:
        raise RuntimeError("RANSAC no encontró un modelo válido.")

    # 6) Re‐estimar H con TODOS los inliers (DLT robusta)
    H_refined = _homography_dlt(pts1[best_inliers], pts2[best_inliers])

    # 7) Métricas finales
    final_errs = _symmetric_transfer_errors(H_refined, pts1, pts2)
    thr2 = 2.0 * (thresh**2)
    final_inliers = final_errs <= thr2
    stats = {
        "iterations": it,
        "inliers": int(np.sum(final_inliers)),
        "total": N,
        "inlier_ratio": float(np.sum(final_inliers)) / float(N),
        "mean_sym_err_inliers": float(np.mean(final_errs[final_inliers])) if np.any(final_inliers) else np.inf,
    }

    return H_refined, final_inliers, stats

# ---------- Error de transferencia simétrico ----------
def _symmetric_transfer_errors(H, pts1, pts2):
    """
    Error simétrico en píxeles^2:
    e = ||x2 - H x1||^2 + ||x1 - H^{-1} x2||^2  (en coordenadas cartesianas)
    Retorna vector (N,)
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    N = pts1.shape[0]

    # x2_hat = H x1
    ones = np.ones((N,1))
    x1h = np.hstack([pts1, ones])
    x2h = np.hstack([pts2, ones])

    Hx1 = (H @ x1h.T).T
    Hx1 = Hx1[:, :2] / Hx1[:, 2:3]
    err1 = np.sum((Hx1 - pts2)**2, axis=1)

    # x1_hat = H^{-1} x2
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        # Si H no es invertible, penalizá fuerte
        return np.full(N, np.inf)
    Hinvx2 = (Hinv @ x2h.T).T
    Hinvx2 = Hinvx2[:, :2] / Hinvx2[:, 2:3]
    err2 = np.sum((Hinvx2 - pts1)**2, axis=1)

    return err1 + err2

def match_sift_indices(desA, desB, ratio=0.75, cross_check=True):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knnAB = bf.knnMatch(desA, desB, k=2)
    ab = [(m.queryIdx, m.trainIdx) for m,n in knnAB if m.distance < ratio*n.distance]
    if not cross_check:
        return np.array(ab, dtype=int)
    knnBA = bf.knnMatch(desB, desA, k=2)
    ba = {(m.queryIdx, m.trainIdx) for m,n in knnBA if m.distance < ratio*n.distance}
    inter = np.array([p for p in ab if (p[1], p[0]) in ba], dtype=int)
    # opcional: orden estable
    if inter.size > 0:
        inter = inter[np.lexsort((inter[:,1], inter[:,0]))]
    return inter

def kp_to_xy(kp):
    # lista de cv2.KeyPoint o array (N,7)-> (N,2)
    if isinstance(kp, np.ndarray):
        return kp[:, :2].astype(np.float64)
    return np.array([k.pt for k in kp], dtype=np.float64)

def build_pts_from_pairs(kpA, kpB, pairs_idx):
    xyA, xyB = kp_to_xy(kpA), kp_to_xy(kpB)
    iA, iB = pairs_idx[:,0], pairs_idx[:,1]
    return xyA[iA], xyB[iB]
