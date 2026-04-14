"""
unir_imagenes.py
Fusiona dos imágenes con solapamiento usando ORB + homografía + blending suave.
El canvas se EXPANDE para acomodar ambas imágenes completas.
"""

import sys
import cv2
import numpy as np


# ─────────────────────────────────────────────
# Constantes configurables
# ─────────────────────────────────────────────
ORB_FEATURES  = 8_000   # Puntos clave máximos para ORB
MIN_MATCHES   = 10      # Mínimo de coincidencias aceptables
BLEND_ZONE_PX = 80      # Tamaño (px) de la transición suave


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────
def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga una imagen BGR; lanza FileNotFoundError si falla."""
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: '{ruta}'")
    return img


# ─────────────────────────────────────────────
# Detección y matching de puntos clave
# ─────────────────────────────────────────────
def detectar_y_emparejar(
    img1: np.ndarray,
    img2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta puntos clave ORB en ambas imágenes y retorna los puntos
    correspondientes como arrays float32.
    """
    orb = cv2.ORB_create(ORB_FEATURES)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("No se detectaron descriptores en una o ambas imágenes.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < MIN_MATCHES:
        raise RuntimeError(
            f"Coincidencias insuficientes: {len(matches)} (mínimo {MIN_MATCHES})."
        )

    matches = sorted(matches, key=lambda m: m.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return pts1, pts2


# ─────────────────────────────────────────────
# Calcular canvas expandido y transformación
# ─────────────────────────────────────────────
def calcular_canvas_expandido(
    img1: np.ndarray,
    img2: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Calcula el tamaño del canvas necesario para contener ambas imágenes
    y devuelve las transformaciones corregidas para cada una.

    Returns
    -------
    H1_offset  : Homografía de img1 al canvas (solo traslación si es necesario)
    H2_final   : Homografía de img2 al canvas
    (ancho, alto) : Dimensiones del canvas resultante
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Esquinas de img1 en su espacio original
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)

    # Esquinas de img2 proyectadas mediante H al espacio de img1
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners2_proj = cv2.perspectiveTransform(corners2, H)

    # Unir todas las esquinas para saber el bounding box total
    all_corners = np.concatenate([corners1, corners2_proj], axis=0)

    x_min = np.floor(all_corners[:, 0, 0].min()).astype(int)
    y_min = np.floor(all_corners[:, 0, 1].min()).astype(int)
    x_max = np.ceil(all_corners[:, 0, 0].max()).astype(int)
    y_max = np.ceil(all_corners[:, 0, 1].max()).astype(int)

    # Offset para que no haya coordenadas negativas
    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0

    ancho  = x_max - x_min
    alto   = y_max - y_min

    # Matriz de traslación para img1
    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1        ],
    ], dtype=np.float64)

    H1_offset = T                    # img1 solo se traslada
    H2_final  = T @ H               # img2: primero H, luego la traslación

    return H1_offset, H2_final, (ancho, alto)


# ─────────────────────────────────────────────
# Composición con blending suave
# ─────────────────────────────────────────────
def componer_canvas(
    img1: np.ndarray,
    img2: np.ndarray,
    H1: np.ndarray,
    H2: np.ndarray,
    tamaño: tuple[int, int],
    zona: int = BLEND_ZONE_PX,
) -> np.ndarray:
    """
    Proyecta ambas imágenes al canvas y las combina con blending
    en la zona de solapamiento.
    """
    ancho, alto = tamaño

    # Proyectar cada imagen al canvas
    warped1 = cv2.warpPerspective(img1, H1, (ancho, alto))
    warped2 = cv2.warpPerspective(img2, H2, (ancho, alto))

    # Máscaras de píxeles válidos (no negros)
    mask1 = (cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    mask2 = (cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

    # Zona de solapamiento
    overlap = mask1 * mask2  # 1 donde ambas tienen contenido

    # ── Blending suave con distancia a los bordes ──────────────────────────
    # Para cada imagen, calculamos la distancia transform dentro de su máscara.
    # En el solapamiento, el peso es proporcional a cuánto "domina" cada imagen.
    dist1 = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)

    # En zonas sin solapamiento cada imagen manda; en solapamiento → mezcla
    total = dist1 + dist2
    total[total == 0] = 1  # evitar división por cero

    alpha1 = dist1 / total  # shape (H, W)
    alpha2 = dist2 / total

    # Expandir a 3 canales
    alpha1 = alpha1[:, :, np.newaxis]
    alpha2 = alpha2[:, :, np.newaxis]

    w1f = warped1.astype(np.float32)
    w2f = warped2.astype(np.float32)

    # Mezcla ponderada
    resultado = w1f * alpha1 + w2f * alpha2

    # Recortar bordes del canvas que estén completamente vacíos
    combined_mask = np.clip(mask1 + mask2, 0, 1).astype(np.uint8)
    coords = cv2.findNonZero(combined_mask)
    x, y, w, h = cv2.boundingRect(coords)
    resultado = resultado[y:y+h, x:x+w]

    return np.clip(resultado, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────
def unir_imagenes(
    ruta_img1: str,
    ruta_img2: str,
    ruta_salida: str = "resultado.jpg",
) -> None:
    """
    Fusiona dos imágenes con solapamiento y guarda el resultado
    en un canvas expandido que contiene ambas imágenes completas.

    Parameters
    ----------
    ruta_img1   : Imagen de referencia.
    ruta_img2   : Imagen a alinear y unir.
    ruta_salida : Ruta del archivo de salida.
    """
    # 1. Carga
    img1 = cargar_imagen(ruta_img1)
    img2 = cargar_imagen(ruta_img2)

    # 2. Detectar puntos clave y emparejarlos
    pts1, pts2 = detectar_y_emparejar(img1, img2)

    # 3. Calcular homografía de img2 → espacio de img1
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None:
        raise RuntimeError("No se pudo calcular una homografía válida.")

    inliers = int(mask.sum())
    print(f"Homografía calculada con {inliers} inliers.")

    # 4. Calcular canvas expandido y homografías finales
    H1, H2, tamaño = calcular_canvas_expandido(img1, img2, H)
    print(f"Canvas resultante: {tamaño[0]} × {tamaño[1]} px")

    # 5. Componer con blending suave
    resultado = componer_canvas(img1, img2, H1, H2, tamaño)

    # 6. Guardar
    cv2.imwrite(ruta_salida, resultado)
    print(f"Imagen guardada en: {ruta_salida}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 4:
        # Uso: python unir_imagenes.py img1.jpg img2.jpg salida.jpg
        ruta1, ruta2, salida = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        ruta1, ruta2, salida = "arriba.jpg", "abajo.jpg", "resultado.jpg"

    try:
        unir_imagenes(ruta1, ruta2, salida)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

def unir_imagenes_bytes(img1_bytes, img2_bytes):


    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    pts1, pts2 = detectar_y_emparejar(img1, img2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None:
        raise RuntimeError("No se pudo calcular homografía")

    H1, H2, tamaño = calcular_canvas_expandido(img1, img2, H)
    resultado = componer_canvas(img1, img2, H1, H2, tamaño)

    _, buffer = cv2.imencode(".jpg", resultado)
    return buffer.tobytes()