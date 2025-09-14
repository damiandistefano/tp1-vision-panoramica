import cv2
import numpy as np
import matplotlib.pyplot as plt


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


    return kp, responses , des



def anms(keypoints, responses, N):
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

    return keypoints_sel, responses_sel, selected




def pick_points_cv(img, win_name="Pick 4 points", n_points=4, out_npy=None):
   
    points = []

    img= img.copy()
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


# # ---------- RANSAC ----------
# def ransac_homography(pts1, pts2, thresh=3.0, max_iters=2000, confidence=0.999, random_state=None):
#     """
#     Estima H con RANSAC sin OpenCV.
#     pts1, pts2: (N,2) correspondencias
#     thresh: umbral de inliers en píxeles (error simétrico <= thresh^2*2 aprox)
#     max_iters: tope duro
#     confidence: prob. de tener al menos una muestra libre de outliers (ajusta iteraciones adaptativamente)
#     return: H_best (3x3), inlier_mask (N,bool), stats (dict)
#     """
#     rng = np.random.default_rng(random_state)
#     pts1 = np.asarray(pts1, dtype=np.float64)
#     pts2 = np.asarray(pts2, dtype=np.float64)
#     N = pts1.shape[0]
#     if N < 4:
#         raise ValueError("Se requieren al menos 4 correspondencias.")

#     best_H = None
#     best_inliers = None
#     best_inlier_count = 0

#     s = 4  # tamaño de muestra mínima para homografía
#     # Iteraciones adaptativas (se actualiza cuando mejora w)
#     it = 0
#     max_adaptive = max_iters

#     while it < max_adaptive:
#         it += 1
#         # 1) Muestreo mínimo sin reemplazo
#         idx = rng.choice(N, size=s, replace=False)
#         try:
#             H = _homography_dlt(pts1[idx], pts2[idx])
#         except np.linalg.LinAlgError:
#             continue  # muestra degenerada (colineal, etc.)

#         # 2) Medir errores para todos
#         errs = _symmetric_transfer_errors(H, pts1, pts2)

#         # 3) Inliers
#         # Comparación con thresh en píxeles: usamos err <= 2*thresh^2 (dos proyecciones)
#         thr2 = 2.0 * (thresh**2)
#         inliers = errs <= thr2
#         count = int(np.sum(inliers))

#         # 4) Actualizar mejor
#         if count > best_inlier_count:
#             best_inlier_count = count
#             best_inliers = inliers
#             best_H = H

#             # 5) Actualizar nº de iteraciones necesarias (adaptativo)
#             w = count / float(N)
#             w = np.clip(w, 1e-6, 1-1e-6)
#             num = np.log(1 - confidence)
#             den = np.log(1 - (w**s))
#             max_adaptive = min(max_iters, int(np.ceil(num / den)))
#             if max_adaptive <= it:  # ya alcanzamos la confianza deseada
#                 break

#     if best_H is None:
#         raise RuntimeError("RANSAC no encontró un modelo válido.")

#     # 6) Re‐estimar H con TODOS los inliers (DLT robusta)
#     H_refined = _homography_dlt(pts1[best_inliers], pts2[best_inliers])

#     # 7) Métricas finales
#     final_errs = _symmetric_transfer_errors(H_refined, pts1, pts2)
#     thr2 = 2.0 * (thresh**2)
#     final_inliers = final_errs <= thr2
#     stats = {
#         "iterations": it,
#         "inliers": int(np.sum(final_inliers)),
#         "total": N,
#         "inlier_ratio": float(np.sum(final_inliers)) / float(N),
#         "mean_sym_err_inliers": float(np.mean(final_errs[final_inliers])) if np.any(final_inliers) else np.inf,
#     }

#     return H_refined, final_inliers, stats

# # ---------- Error de transferencia simétrico ----------
# def _symmetric_transfer_errors(H, pts1, pts2):
#     """
#     Error simétrico en píxeles^2:
#     e = ||x2 - H x1||^2 + ||x1 - H^{-1} x2||^2  (en coordenadas cartesianas)
#     Retorna vector (N,)
#     """
#     pts1 = np.asarray(pts1, dtype=np.float64)
#     pts2 = np.asarray(pts2, dtype=np.float64)
#     N = pts1.shape[0]

#     # x2_hat = H x1
#     ones = np.ones((N,1))
#     x1h = np.hstack([pts1, ones])
#     x2h = np.hstack([pts2, ones])

#     Hx1 = (H @ x1h.T).T
#     Hx1 = Hx1[:, :2] / Hx1[:, 2:3]
#     err1 = np.sum((Hx1 - pts2)**2, axis=1)

#     # x1_hat = H^{-1} x2
#     try:
#         Hinv = np.linalg.inv(H)
#     except np.linalg.LinAlgError:
#         # Si H no es invertible, penalizá fuerte
#         return np.full(N, np.inf)
#     Hinvx2 = (Hinv @ x2h.T).T
#     Hinvx2 = Hinvx2[:, :2] / Hinvx2[:, 2:3]
#     err2 = np.sum((Hinvx2 - pts1)**2, axis=1)

#     return err1 + err2



def _normalize_points(pts):
    """
    Normaliza puntos 2D (Nx2) de forma isotrópica (traslación + escala) para DLT.
    Devuelve (T, pts_norm) donde T es la matriz 3x3 de normalización y pts_norm son Nx2.
    """
    mean = pts.mean(axis=0)
    pts_centered = pts - mean
    mean_dist = np.mean(np.sqrt(np.sum(pts_centered**2, axis=1)))
    if mean_dist < 1e-9:
        scale = 1.0
    else:
        scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0,    0,               1.0]
    ], dtype=np.float64)
    # aplicar T
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = pts_norm_h[:, :2] / pts_norm_h[:, 2:3]
    return T, pts_norm

def _homography_dlt(pts1, pts2):
    """
    Estima homografía H (3x3) que mapea pts1 -> pts2 usando DLT con normalización.
    pts1, pts2: Nx2 arrays con N >= 4.
    Devuelve H con H[2,2] == 1 (si es posible); si falla devuelve None.
    """
    n = pts1.shape[0]
    
    T1, p1n = _normalize_points(pts1)
    T2, p2n = _normalize_points(pts2)

    A = []
    for i in range(n):
        x, y = p1n[i,0], p1n[i,1]
        u, v = p2n[i,0], p2n[i,1]
        A.append([-x, -y, -1,   0,  0,  0, x*u, y*u, u])
        A.append([ 0,  0,  0,  -x, -y, -1, x*v, y*v, v])
    A = np.array(A, dtype=np.float64)
    # SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape(3,3)
    # Des-normalizar: H = T2^{-1} * Hn * T1
    H = np.linalg.inv(T2) @ Hn @ T1
    # Normalizar para que H[2,2] == 1 si posible
    if np.abs(H[2,2]) < 1e-12:
        H = H / (np.linalg.norm(H))
    else:
        H = H / H[2,2]
    return H

def _reprojection_errors(H, pts1, pts2):
    """
    Calcula la distancia euclidiana (reprojection error) entre pts2 y H*pts1.
    Devuelve array de tamaño N con errores.
    """
    n = pts1.shape[0]
    ones = np.ones((n,1), dtype=np.float64)
    p1h = np.hstack([pts1, ones])
    proj_h = (H @ p1h.T).T  # Nx3
    proj_xy = proj_h[:, :2] / proj_h[:, 2:3]
    errs = np.linalg.norm(proj_xy - pts2, axis=1)
    return errs

def ransac_homography(pts1,pts2,n_iter = 2000,reproj_thresh = 4.0,
    min_inliers = 4,stop_inlier_ratio = 0.995,
    random_seed = 42,refine = True) :
    """
    RANSAC para homografía: estima H que mapea pts1 -> pts2.

    Args:
        pts1, pts2: arrays (N,2) float, con correspondencias punto a punto.
        n_iter: número máximo de iteraciones RANSAC.
        reproj_thresh: umbral (en píxeles) para considerar un inlier por reproyección.
        min_inliers: mínimo de inliers aceptable para considerar resultado válido.
        stop_inlier_ratio: si se alcanza esta fracción de inliers respecto N termina antes.
        random_seed: semilla aleatoria (opcional).
        refine: si True, recomputa H usando *todos* los inliers finales (DLT).

    Returns: diccionario con:
        - 'H': homografía 3x3 (o None si no se encontró modelo válido)
        - 'mask': array uint8 (N,) con 1 para inlier, 0 para outlier (si H is None -> zeros)
        - 'inliers_idx': lista de índices inliers
        - 'inlier_ratio': fracción de inliers (float)
        - 'reproj_errors': array (N,) con errores de reproyección (inf si H None)
        - 'num_inliers': número de inliers (int)
        - 'iterations': iteraciones realmente usadas (int)
        - 'status': 'ok' or 'not_enough_inliers'
    """

    N = pts1.shape[0]
    rng = np.random.default_rng(random_seed)

    best_H = None
    best_inliers = np.zeros((N,), dtype=bool)
    best_count = 0
    best_errors = np.full((N,), np.inf, dtype=np.float64)
    iterations_used = 0

    for it in range(n_iter):
        iterations_used += 1
        # muestreo aleatorio de 4 indices distintos
        idx = rng.choice(N, size=4, replace=False)
    
        H_candidate = _homography_dlt(pts1[idx], pts2[idx])
        

        errs = _reprojection_errors(H_candidate, pts1, pts2)
        inliers = errs <= reproj_thresh
        count = int(inliers.sum())

        # actualizar mejor modelo
        if count > best_count:
            best_count = count
            best_H = H_candidate
            best_inliers = inliers
            best_errors = errs

            # early stop si alcanzamos umbral
            if (best_count / N) >= stop_inlier_ratio:
                break

    # comprobar resultado
    if best_H is None or best_count < min_inliers:
        return {
            'H': None,
            'mask': np.zeros((N,), dtype=np.uint8),
            'inliers_idx': [],
            'inlier_ratio': 0.0,
            'reproj_errors': np.full((N,), np.inf, dtype=np.float64),
            'num_inliers': 0,
            'iterations': iterations_used,
            'status': 'not_enough_inliers'
        }

    # Refinar con todos los inliers si se pide
    if refine:
        inlier_idxs = np.nonzero(best_inliers)[0]
        H_refined = _homography_dlt(pts1[inlier_idxs], pts2[inlier_idxs])
        # recalcular errores con H_refined
        final_errors = _reprojection_errors(H_refined, pts1, pts2)
        final_inliers = final_errors <= reproj_thresh
        # si el refinado empeora drásticamente el número de inliers, mantener el anterior
        if final_inliers.sum() >= max(min_inliers, int(0.5 * best_count)):
            best_H = H_refined
            best_inliers = final_inliers
            best_errors = final_errors


    mask = best_inliers.astype(np.uint8)
    inliers_idx = list(np.nonzero(best_inliers)[0])
    inlier_ratio = mask.sum() / N
    result = {
        'H': best_H,
        'mask': mask,
        'inliers_idx': inliers_idx,
        'inlier_ratio': float(inlier_ratio),
        'reproj_errors': best_errors,
        'num_inliers': int(mask.sum()),
        'iterations': iterations_used,
        'status': 'ok'
    }
    return result

























