"""
Forensic Dossier & Case Report Generator
========================================
Compiles official police and forensic incident reports for identified suspects and surveillance events.
Outputs printable Markdown, HTML, and structured JSON dossiers.
"""

import time
import json
from typing import Dict, Any, List, Optional


class ForensicReportGenerator:
    """
    Generates official law enforcement intelligence dossiers.
    """

    @staticmethod
    def generate_case_dossier_markdown(
        incident_id: str,
        suspect_name: str,
        criminal_id: str,
        similarity_score: float,
        camera_id: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dwell_seconds: float = 0.0
    ) -> str:
        """
        Creates an official incident report in Markdown format.
        """
        ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
        meta = metadata or {}
        confidence_pct = round(similarity_score * 100, 1)

        md = f"""# 🏛️ STATE POLICE DEPARTMENT — FORENSIC IDENTIFICATION DOSSIER
**Incident Case Reference:** `{incident_id}`  
**Classification:** RESTRICTED / LAW ENFORCEMENT SENSITIVE  
**Generated At:** `{ts}`  

---

## 1. Suspect Profile & Identification Summary

| Field | Detail |
|---|---|
| **Full Legal Name** | **{suspect_name}** |
| **Criminal Registry ID** | `{criminal_id}` |
| **Neural Match Confidence** | **{confidence_pct}%** (ArcFace 512-dim Cosine Metric) |
| **Alias / Street Names** | {meta.get('alias', 'N/A')} |
| **Crime Classification** | {meta.get('crime_history', 'N/A')} |

---

## 2. Surveillance Sighting & Telemetry

| Parameter | Value |
|---|---|
| **Camera Feed Source** | `{camera_id}` |
| **Timestamp of Sighting** | `{ts}` |
| **Dwell Duration in Frame** | `{dwell_seconds:.1f} seconds` |
| **Neural Engine Model** | InsightFace SCRFD 10G + ArcFace ResNet-50 |

---

## 3. Forensic Chain of Custody & Verification

1. **Biometric Feature Consistency**: 512-dimensional vector matched against local encrypted ChromaDB repository.
2. **Quality & Pose Normalization**: 5-Point Affine canonical warp and LAB-CLAHE enhancement applied prior to inference.
3. **Automated SOS Notification**: Dispatched to field patrol dispatch channels.

---

### Officer Sign-Off & Review
- **Investigating Officer:** _______________________________
- **Badge / PIN Number:** _________________________________
- **Signature:** _________________________ **Date:** ______
"""
        return md
