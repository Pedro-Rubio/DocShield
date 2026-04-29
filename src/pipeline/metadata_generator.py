"""
Módulo de generación y síntesis de metadatos de sesión.

Genera features basadas en el contexto de captura del documento:
dispositivo, IP, intentos previos, y señales de liveness.
Estos metadatos se usan como features adicionales para la
clasificación de fraude.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def generate_session_metadata(
    capture_meta: dict,
    user_id: Optional[str] = None,
    attempt_history: Optional[list[dict]] = None,
) -> dict:
    """
    Genera metadatos de sesión a partir de datos de captura.

    Args:
        capture_meta: Diccionario con metadatos de captura. Se espera:
            - user_agent (str): User-Agent del dispositivo
            - screen_width (int): Ancho de pantalla
            - screen_height (int): Alto de pantalla
            - platform (str): Plataforma (iOS, Android, web)
            - ip_address (str): Dirección IP del usuario
            - liveness_passed (bool): Si pasó la prueba de liveness
            - accelerometer_data (list): Datos del acelerómetro
        user_id: Identificador único del usuario (opcional).
        attempt_history: Historial de intentos previos (opcional).

    Returns:
        Diccionario con las siguientes keys:
        - ip_risk_score: Riesgo estimado de la IP (0-1)
        - emulator_detected: Si se detectó emulador (0 o 1)
        - tor_detected: Si se detectó uso de Tor (0 o 1)
        - vpn_detected: Si se detectó uso de VPN (0 o 1)
        - repeated_attempts: Número de intentos previos
        - liveness_passed: Resultado de liveness (0 o 1)
        - device_fingerprint_score: Score de fingerprint del dispositivo
    """
    ip_risk_score = _compute_ip_risk(capture_meta.get("ip_address", ""))
    emulator_detected = _detect_emulator(capture_meta)
    tor_detected = _detect_tor(capture_meta.get("user_agent", ""))
    vpn_detected = _detect_vpn(capture_meta.get("user_agent", ""))
    repeated_attempts = _count_repeated_attempts(attempt_history)
    liveness_passed = _evaluate_liveness(capture_meta)
    device_fingerprint_score = _compute_device_fingerprint(capture_meta)

    return {
        "ip_risk_score": ip_risk_score,
        "emulator_detected": emulator_detected,
        "tor_detected": tor_detected,
        "vpn_detected": vpn_detected,
        "repeated_attempts": repeated_attempts,
        "liveness_passed": liveness_passed,
        "device_fingerprint_score": device_fingerprint_score,
    }


def _compute_ip_risk(ip_address: str) -> float:
    """
    Calcula el score de riesgo de una dirección IP.

    Evalúa si la IP proviene de:
    - Proxy/VPN conocido (alto riesgo)
    - Tor exit node (muy alto riesgo)
    - Datacenter (medio riesgo)
    - ISP residencial (bajo riesgo)

    Args:
        ip_address: Dirección IP del usuario.

    Returns:
        Score de riesgo entre 0 (bajo) y 1 (alto).
    """
    if not ip_address:
        return 0.5  # Valor neutral si no hay IP

    # En producción, esto consultaría una API como MaxMind, IP2Location, etc.
    # Aquí usamos heurísticas básicas:

    # IPs privadas/loopback = riesgo medio (posible testing)
    if ip_address.startswith(("10.", "172.", "192.168.", "127.", "0.")):
        return 0.3

    # Simulación: en producción usar base de datos de IPs
    # Por ahora retornar valor neutro
    return 0.1


def _detect_emulator(capture_meta: dict) -> int:
    """
    Detecta si la captura viene de un emulador.

    Señales de emulación:
    - User-Agent contiene "emulator", "simulator", "android SDK"
    - Resolución inusual para un dispositivo móvil
    - Ausencia de datos de sensores reales

    Args:
        capture_meta: Metadatos de captura.

    Returns:
        1 si se detecta emulador, 0 en caso contrario.
    """
    ua = capture_meta.get("user_agent", "").lower()
    emulator_signals = ["emulator", "simulator", "sdk", "genymotion", "bluestacks"]

    if any(signal in ua for signal in emulator_signals):
        return 1

    # Verificar si hay datos de acelerómetro reales
    accel = capture_meta.get("accelerometer_data")
    if accel is None or len(accel) == 0:
        return 1  # Sin datos de sensor = posible emulador

    # Verificar que el acelerómetro tenga variación real
    accel_values = np.array(accel)
    if np.std(accel_values) < 0.001:
        return 1  # Datos estáticos = posible emulador

    return 0


def _detect_tor(user_agent: str) -> int:
    """
    Detecta posible uso de la red Tor.

    En producción se usaría una lista de exit nodes de Tor.
    Aquí usamos señales básicas del user-agent.

    Args:
        user_agent: User-Agent del cliente.

    Returns:
        1 si se detecta Tor, 0 en caso contrario.
    """
    # En producción: consultar la lista de exit nodes de Tor
    # https://check.torproject.org/
    # Por ahora: señal básica
    ua = user_agent.lower()
    if "tor" in ua or "torbrowser" in ua:
        return 1
    return 0


def _detect_vpn(user_agent: str) -> int:
    """
    Detecta posible uso de VPN.

    En producción se usaría una base de datos de rangos IP de VPNs.

    Args:
        user_agent: User-Agent del cliente.

    Returns:
        1 si se detecta VPN, 0 en caso contrario.
    """
    ua = user_agent.lower()
    vpn_signals = ["vpn", "expressvpn", "nordvpn", "cyberghost"]

    if any(signal in ua for signal in vpn_signals):
        return 1
    return 0


def _count_repeated_attempts(attempt_history: Optional[list[dict]]) -> int:
    """
    Cuenta el número de intentos previos del mismo usuario.

    Args:
        attempt_history: Lista de intentos previos.

    Returns:
        Número de intentos previos.
    """
    if attempt_history is None:
        return 0
    return len(attempt_history)


def _evaluate_liveness(capture_meta: dict) -> int:
    """
    Evalúa si la prueba de liveness fue exitosa.

    La prueba requiere que el usuario incline el documento
    ±10° usando el acelerómetro del teléfono dentro de 8 segundos.

    Args:
        capture_meta: Metadatos de captura con datos del acelerómetro.

    Returns:
        1 si liveness pasó, 0 en caso contrario.
    """
    liveness_passed = capture_meta.get("liveness_passed", False)
    if isinstance(liveness_passed, bool):
        return 1 if liveness_passed else 0
    return 0


def _compute_device_fingerprint(capture_meta: dict) -> float:
    """
    Calcula un score de fingerprint del dispositivo.

    Combina:
    - Resolución de pantalla
    - Plataforma
    - User-Agent

    Args:
        capture_meta: Metadatos de captura.

    Returns:
        Score entre 0 y 1 indicando cuán confiable es el fingerprint.
    """
    screen_w = capture_meta.get("screen_width", 0)
    screen_h = capture_meta.get("screen_height", 0)
    platform = capture_meta.get("platform", "").lower()

    score = 0.5  # Base neutral

    # Resolución razonable para móvil aumenta confianza
    if screen_w > 0 and screen_h > 0:
        if 320 <= screen_w <= 2560 and 480 <= screen_h <= 2560:
            score += 0.2

    # Plataforma conocida aumenta confianza
    if platform in ("ios", "android"):
        score += 0.2

    return min(score, 1.0)
