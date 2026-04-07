"""
Severity grading service for pneumonia detection.
Evaluates pneumonia severity based on model confidence and provides
diagnostic recommendations in Uzbek language for clinicians.
"""

from typing import Dict, List


def grade_severity(confidence: float, prob_pneumonia: float) -> Dict:
    """
    Pnevmoniya darajasini aniqlash.

    Args:
        confidence: Model ishonch darajasi (0-1)
        prob_pneumonia: Pnevmoniya ehtimolligi (0-1)

    Returns:
        Dict with severity level, description, color, and recommendations
    """
    pct = prob_pneumonia * 100

    if prob_pneumonia < 0.50:
        return {
            "severity": "normal",
            "severity_level": 0,
            "severity_description": "Normal — pnevmoniya belgilari aniqlanmadi",
            "severity_color": "#10b981",
            "affected_area_percent": 0.0,
            "recommendations": [
                "O'pka tasvirida pnevmoniyaga xos o'zgarishlar kuzatilmadi.",
                "Bemor umumiy holati yaxshi bo'lsa, qo'shimcha tekshiruv shart emas.",
                "Profilaktik tibbiy ko'rikdan o'tish tavsiya etiladi.",
            ],
        }

    if prob_pneumonia < 0.70:
        return {
            "severity": "yengil",
            "severity_level": 1,
            "severity_description": f"Yengil daraja — pnevmoniya ehtimoli {pct:.0f}%",
            "severity_color": "#f59e0b",
            "affected_area_percent": round(prob_pneumonia * 25, 1),
            "recommendations": [
                "O'pkada yengil darajali yallig'lanish belgilari aniqlandi.",
                "Qo'shimcha laboratoriya tekshiruvlari (qon tahlili, CRP) o'tkazish tavsiya etiladi.",
                "Antibiotik terapiyasini boshlash to'g'risida mutaxassis bilan maslahat qiling.",
                "2-3 kundan so'ng qayta rentgen tekshiruvidan o'tish kerak.",
                "Bemorga dam olish va ko'p suyuqlik ichish tavsiya etilsin.",
            ],
        }

    if prob_pneumonia < 0.85:
        return {
            "severity": "o'rta",
            "severity_level": 2,
            "severity_description": f"O'rta daraja — pnevmoniya ehtimoli {pct:.0f}%",
            "severity_color": "#f97316",
            "affected_area_percent": round(prob_pneumonia * 45, 1),
            "recommendations": [
                "O'pkada o'rta darajadagi pnevmoniya belgilari aniqlandi.",
                "ZUDLIK bilan pulmonolog konsultatsiyasi talab etiladi.",
                "Keng spektrli antibiotik terapiyasini boshlash kerak.",
                "To'liq qon tahlili, CRP, prokalsitonin tekshiruvlari zarur.",
                "KT tekshiruvini o'tkazish tavsiya etiladi (aniqroq baho uchun).",
                "Bemorni stasionar kuzatuvga olish ko'rib chiqilsin.",
                "Kislorod saturatsiyasini muntazam nazorat qilish kerak.",
            ],
        }

    return {
        "severity": "og'ir",
        "severity_level": 3,
        "severity_description": f"Og'ir daraja — pnevmoniya ehtimoli {pct:.0f}%",
        "severity_color": "#ef4444",
        "affected_area_percent": round(prob_pneumonia * 65, 1),
        "recommendations": [
            "O'pkada OG'IR darajadagi pnevmoniya belgilari aniqlandi!",
            "SHOSHILINCH TIBBIY YORDAM — bemorni zudlik bilan stasionarga yotqizish zarur.",
            "Intensiv terapiya bo'limiga yo'naltirish ko'rib chiqilsin.",
            "Kuchli antibiotik kombinatsiyasi (IV) bilan davolashni boshlash kerak.",
            "KT tekshiruvi, qon gazlari tahlili, prokalsitonin, D-dimer zarur.",
            "Kislorod terapiyasini zudlik bilan boshlash kerak.",
            "Mexanik ventilyatsiya ehtiyoji baholansin.",
            "Har 6 soatda vital belgilarni nazorat qilish zarur.",
        ],
    }


def get_severity_summary(severity_data: Dict) -> str:
    """Severity natijasini qisqa matn sifatida qaytarish."""
    return (
        f"Daraja: {severity_data['severity'].upper()} | "
        f"{severity_data['severity_description']} | "
        f"Shikastlangan hudud: ~{severity_data['affected_area_percent']:.0f}%"
    )
