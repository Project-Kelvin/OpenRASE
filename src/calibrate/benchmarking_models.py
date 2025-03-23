"""
This defines the models to calibrate the VNFs.
"""

import os
from typing import Any
import tensorflow as tf
import numpy as np
import random

os.environ["PYTHONHASHSEED"] = "100"

# Setting the seed for numpy-generated random numbers
np.random.seed(100)

# Setting the seed for python random numbers
random.seed(100)

# Setting the graph-level random seed.
tf.random.set_seed(100)


wafCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    )
]

wafMem: "list[Any]" = [
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

wafLatency: "list[Any]" = [
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

tmCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

tmMem: "list[Any]" = [
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    )
]

tmLatency: "list[Any]" = [
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

lbCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=4,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=24,
        activation="relu",
        kernel_initializer="he_normal",
    )
]

lbMemory: "list[Any]" = [
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

lbLatency: "list[Any]" = [
    tf.keras.layers.Dense(
        units=4,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

ipsCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=8,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

ipsMemory: "list[Any]" = [
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal",
    )
]

ipsLatency: "list[Any]" = [
    tf.keras.layers.Dense(
        units=8,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

idsCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=8,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

idsMemory: "list[Any]" = [
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=256,
        activation="relu",
        kernel_initializer="he_normal",
    )
]

idsLatency: "list[Any]" = [
    tf.keras.layers.Dense(
        units=8,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

haCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

haMemory: "list[Any]" = [
    tf.keras.layers.Dense(
        units=150,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=150,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=256,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

haLatency: "list[Any]" = [
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

dpiCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=4,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=4,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

dpiMemory: "list[Any]" = [
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

dpiLatency: "list[Any]" = [
    tf.keras.layers.Dense(
        units=8,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=4,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

dummyCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    )
]

dummyMemory: "list[Any]" = [
    tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal",
    )
]

dummyLatency: "list[Any]" = [
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

vnfModels: "dict[str, dict[str, Any]]" = {
    "waf": {
        "cpu": wafCPU,
        "memory": wafMem,
        "median": wafLatency,
    },
    "tm": {
        "cpu": tmCPU,
        "memory": tmMem,
        "median": tmLatency,
    },
    "lb": {
        "cpu": lbCPU,
        "memory": lbMemory,
        "median": lbLatency,
    },
    "ips": {
        "cpu": ipsCPU,
        "memory": ipsMemory,
        "median": ipsLatency,
    },
    "ids": {
        "cpu": idsCPU,
        "memory": idsMemory,
        "median": idsLatency,
    },
    "ha": {
        "cpu": haCPU,
        "memory": haMemory,
        "median": haLatency,
    },
    "dpi": {
        "cpu": dpiCPU,
        "memory": dpiMemory,
        "median": dpiLatency,
    },
    "dummy": {
        "cpu": dummyCPU,
        "memory": dummyMemory,
        "median": dummyLatency,
    },
}
