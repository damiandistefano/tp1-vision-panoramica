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



def mostrar_matches(img1, kp1, des1, img2, kp2, des2):
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
