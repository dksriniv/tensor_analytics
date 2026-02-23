"""Quick GPU check for Apple Silicon using PyTorch (MPS) and TensorFlow (Metal)."""

import sys
import time

import tensorflow as tf
import torch


def check_tensorflow() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print("TensorFlow: GPU/Metal available.")
            # Lightweight op to exercise the device selection.
            a = tf.constant([1.0, 2.0, 3.0], shape=[1, 3], name="a")
            b = tf.constant([4.0, 5.0, 6.0], shape=[3, 1], name="b")
            c = tf.matmul(a, b)
            print("TensorFlow matmul result:", c.numpy())
        else:
            print("TensorFlow: using CPU (no GPU/Metal detected).")
    except Exception as exc:  # noqa: BLE001
        print(f"TensorFlow error: {exc}")


def check_torch_mps() -> None:
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    print(f"torch mps_is_built: {mps_built}")
    print(f"torch mps_is_available: {mps_available}")
    if mps_available:
        try:
            tensor = torch.ones(1, device="mps") * 2
            print(f"torch tensor_device: {tensor.device} value: {tensor.item()}")
        except Exception as exc:  # noqa: BLE001
            print(f"torch MPS runtime error: {exc}")
    else:
        print("torch: MPS not available; using CPU.")


def run_torch_matmul() -> None:
    """Run a small matmul on MPS if available."""
    if not torch.backends.mps.is_available():
        print("torch matmul: skipped (MPS not available).")
        return
    try:
        a = torch.randn(256, 256, device="mps")
        b = torch.randn(256, 256, device="mps")
        c = a @ b
        print(f"torch matmul device: {c.device} sum: {c.sum().item():.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch matmul error: {exc}")


def run_tensorflow_matmul() -> None:
    """Run a small matmul on GPU/Metal if available."""
    try:
        if not tf.config.list_physical_devices("GPU"):
            print("TensorFlow matmul: skipped (no GPU/Metal detected).")
            return
        with tf.device("/GPU:0"):
            a = tf.random.normal([256, 256])
            b = tf.random.normal([256, 256])
            c = tf.matmul(a, b)
            print("TensorFlow matmul device: /GPU:0 sum:", float(tf.reduce_sum(c)))
    except Exception as exc:  # noqa: BLE001
        print(f"TensorFlow matmul error: {exc}")


def compare_torch_perf(size: int = 1024, reps: int = 3) -> None:
    """Compare a heavier matmul+relu+matmul workload on CPU vs MPS (if available)."""
    if not torch.backends.mps.is_available():
        print("torch perf: skipped (MPS not available).")
        return

    def _matmul(dev: torch.device) -> float:
        a = torch.randn(size, size, device=dev)
        b = torch.randn(size, size, device=dev)
        bias = torch.randn(size, size, device=dev)

        def _work():
            c = torch.relu(a @ b + bias)
            d = c @ b
            return d

        # Warmup
        _ = _work()
        if dev.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            c = _work()
        if dev.type == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()
        # prevent optimizations
        _ = c.sum().item()
        return (end - start) / reps

    cpu_time = _matmul(torch.device("cpu"))
    mps_time = _matmul(torch.device("mps"))
    faster = "mps" if mps_time < cpu_time else "cpu"
    ratio = cpu_time / mps_time if mps_time else float("inf")
    print(
        f"torch perf (matmul+relu+matmul, size={size}, reps={reps}): cpu {cpu_time:.6f}s vs mps {mps_time:.6f}s "
        f"(faster: {faster}, speedup: {ratio:.2f}x vs mps)"
    )


def compare_tensorflow_perf(size: int = 1024, reps: int = 3) -> None:
    """Compare a heavier matmul+relu+matmul workload on CPU vs GPU/Metal if available."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("TensorFlow perf: skipped (no GPU/Metal detected).")
        return

    def _matmul(device: str) -> float:
        with tf.device(device):
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            bias = tf.random.normal([size, size])

            def _work():
                c = tf.nn.relu(tf.matmul(a, b) + bias)
                d = tf.matmul(c, b)
                return d

            # Warmup
            _ = _work()
            start = time.perf_counter()
            for _ in range(reps):
                c = _work()
            # Force sync
            _ = c.numpy().sum()
            end = time.perf_counter()
            return (end - start) / reps

    cpu_time = _matmul("/CPU:0")
    gpu_time = _matmul("/GPU:0")
    faster = "gpu" if gpu_time < cpu_time else "cpu"
    ratio = cpu_time / gpu_time if gpu_time else float("inf")
    print(
        f"TensorFlow perf (matmul+relu+matmul, size={size}, reps={reps}): cpu {cpu_time:.6f}s vs gpu {gpu_time:.6f}s "
        f"(faster: {faster}, speedup: {ratio:.2f}x vs gpu)"
    )


def main() -> int:
    check_tensorflow()
    check_torch_mps()
    run_tensorflow_matmul()
    run_torch_matmul()
    compare_tensorflow_perf()
    compare_torch_perf()
    return 0


if __name__ == "__main__":
    sys.exit(main())
