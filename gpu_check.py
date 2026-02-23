"""Quick GPU check for Apple Silicon using PyTorch (MPS) and TensorFlow (Metal)."""

import sys

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


def main() -> int:
    check_tensorflow()
    check_torch_mps()
    run_tensorflow_matmul()
    run_torch_matmul()
    return 0


if __name__ == "__main__":
    sys.exit(main())
