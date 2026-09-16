from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class IMDriver(DriverBase):
    @staticmethod
    def is_active():
        return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty

    def get_current_target(self):
        # arch="hbm-pim", num_banks=16 (default HBM-PIM config)
        return GPUTarget("im", "hbm-pim", 16)

    def get_active_torch_device(self):
        raise RuntimeError("IM driver is compile-only in this workspace")

    def get_benchmarker(self):
        raise RuntimeError("IM benchmarker is not available")
