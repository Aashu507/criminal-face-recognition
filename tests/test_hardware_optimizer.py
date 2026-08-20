"""
Unit tests for HardwareOptimizer module.
"""

import pytest
from core.hardware_optimizer import HardwareOptimizer


def test_hardware_telemetry():
    telemetry = HardwareOptimizer.get_system_telemetry()
    assert "cpu_percent" in telemetry
    assert "ram_total_gb" in telemetry
    assert "ram_used_gb" in telemetry
    assert telemetry["ram_total_gb"] > 0
    assert telemetry["status"] in ["HEALTHY", "HIGH_LOAD"]


def test_cuda_options():
    opt = HardwareOptimizer(target_gpu_id=0, vram_limit_mb=2048)
    cuda_opts = opt.get_cuda_provider_options()
    assert cuda_opts["device_id"] == "0"
    assert "gpu_mem_limit" in cuda_opts


def test_tensorrt_options():
    opt = HardwareOptimizer(target_gpu_id=0)
    trt_opts = opt.get_tensorrt_provider_options()
    assert trt_opts["trt_fp16_enable"] == "1"
    assert "trt_max_workspace_size" in trt_opts
