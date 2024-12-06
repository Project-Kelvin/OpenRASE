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
        units=256,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

tmCPU: "list[Any]" = [
    tf.keras.layers.Dense(
        units=15,
        activation="relu",
        kernel_initializer="he_normal",
    )
]

tmMem: "list[Any]" = [
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
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

lbCPU: "list[Any]" = [
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

lbMemory: "list[Any]" = [
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
        units=64,
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
        units=8,
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
        units=128,
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
        units=8,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

idsMemory: "list[Any]" = [
    tf.keras.layers.Dense(
        units=4,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
    tf.keras.layers.Dense(
        units=32,
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
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
    ),
]

haMemory: "list[Any]" = [
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
    ),
]

dpiCPU: "list[Any]" = [
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

dpiMemory: "list[Any]" = [
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

vnfModels: "dict[str, dict[str, Any]]" = {
    "waf": {
        "cpu": wafCPU,
        "memory": wafMem,
    },
    "tm": {
        "cpu": tmCPU,
        "memory": tmMem,
    },
    "lb": {
        "cpu": lbCPU,
        "memory": lbMemory,
    },
    "ips": {
        "cpu": ipsCPU,
        "memory": ipsMemory,
    },
    "ids": {
        "cpu": idsCPU,
        "memory": idsMemory,
    },
    "ha": {
        "cpu": haCPU,
        "memory": haMemory,
    },
    "dpi": {
        "cpu": dpiCPU,
        "memory": dpiMemory,
    },
}
