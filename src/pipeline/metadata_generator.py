from faker import Faker
from typing import Dict

fake = Faker()

def generate_session_metadata() -> Dict[str, any]:
    """
    Genera metadatos simulados de sesión para detección de fraude.

    Returns:
        Diccionario con metadatos: ip_risk_score, emulator_detected,
        tor_detected, vpn_detected, repeated_attempts, liveness_passed.
    """
    return {
        "ip_risk_score": fake.random.uniform(0.0, 1.0),
        "emulator_detected": fake.boolean(chance_of_getting_true=10),
        "tor_detected": fake.boolean(chance_of_getting_true=5),
        "vpn_detected": fake.boolean(chance_of_getting_true=15),
        "repeated_attempts": fake.random_int(min=0, max=5),
        "liveness_passed": fake.boolean(chance_of_getting_true=80)
    }
