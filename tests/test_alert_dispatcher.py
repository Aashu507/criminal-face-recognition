"""
Unit tests for AlertDispatcher module (rate limiting, cooldown, incident recording).
"""

import time
import numpy as np
import pytest
from core.alert_dispatcher import AlertDispatcher


def test_alert_dispatcher_cooldown():
    dispatcher = AlertDispatcher(cooldown_seconds=30.0)

    # First dispatch should succeed
    incident1 = dispatcher.dispatch_alert(
        criminal_id="CRIM-001",
        criminal_name="Wanted Suspect",
        similarity=0.85,
        camera_id="CAM-NORTH"
    )
    assert incident1 is not None
    assert incident1.criminal_id == "CRIM-001"
    assert incident1.camera_id == "CAM-NORTH"

    # Immediate second dispatch for same criminal should be throttled (return None)
    incident2 = dispatcher.dispatch_alert(
        criminal_id="CRIM-001",
        criminal_name="Wanted Suspect",
        similarity=0.88,
        camera_id="CAM-NORTH"
    )
    assert incident2 is None

    # Different criminal should dispatch immediately
    incident3 = dispatcher.dispatch_alert(
        criminal_id="CRIM-002",
        criminal_name="Second Suspect",
        similarity=0.78,
        camera_id="CAM-SOUTH"
    )
    assert incident3 is not None
    assert incident3.criminal_id == "CRIM-002"
    assert len(dispatcher.incident_history) == 2
