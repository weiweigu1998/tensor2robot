# coding=utf-8
"""TF 2.x drop-in shims for the removed ``tensorflow.contrib`` namespace.

``tf.contrib`` was deleted in TensorFlow 2.0. The BC-Z model and its
``tensor2robot`` dependency chain still import from it. This module provides
the ``tf.contrib`` sub-modules actually referenced along the BC-Z
import / train / eval path so the code runs unmodified on TF 2.x.

Design notes:

* ``slim`` / ``layers`` map to the maintained standalone ``tf-slim`` package,
  which is an API-compatible continuation of ``tf.contrib.slim``.
* ``framework`` exposes only the three members the chain uses
  (``nest``, ``TensorSpec``, ``get_variables``).
* TPU-specific symbols (``tf.contrib.tpu.*``), the exotic optimizers in
  ``tf.contrib.opt``, and the export/predictor helpers are **stubs**. The LfO
  benchmark runs BC-Z single-GPU with a hand-written training loop, so those
  code paths are never executed; the stubs exist only so module import
  succeeds. Calling a stubbed symbol raises a clear RuntimeError.

This file is the *only* net-new code added to the tensor2robot fork; every
other change is a one-line import redirect. The BC-Z model definition, layer
ops, losses, and the resnet / FiLM stack are bit-for-bit identical to
upstream master.
"""
import types

import tensorflow.compat.v1 as tf
import tf_slim as _tf_slim

# --- tf.contrib.slim / tf.contrib.layers ------------------------------------
# tf-slim is the maintained drop-in continuation of tf.contrib.slim.
slim = _tf_slim
layers = _tf_slim


# --- tf.contrib.framework ---------------------------------------------------
def _get_variables(scope=None, collection=None):
    collection = collection or tf.GraphKeys.GLOBAL_VARIABLES
    return tf.get_collection(collection, scope=scope)


framework = types.SimpleNamespace(
    nest=tf.nest,
    TensorSpec=tf.TensorSpec,
    get_variables=_get_variables,
    arg_scope=_tf_slim.arg_scope,
)


# --- tf.contrib.training ----------------------------------------------------
# checkpoints_iterator / create_train_op live under tf.train / tf.compat.v1.
training = tf.train


# --- stubs: TPU, exotic optimizers, export/predictor helpers ----------------
def _stub(qualified_name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"{qualified_name} is a TF1 tf.contrib symbol with no TF2 "
            "equivalent; it is stubbed because this code path is not used "
            "in the LfO benchmark's single-GPU BC-Z setup.")
    return _raise


tpu = types.SimpleNamespace(
    RunConfig=type("RunConfig", (), {}),
    TPUConfig=type("TPUConfig", (), {}),
    TPUEstimator=_stub("tf.contrib.tpu.TPUEstimator"),
    TPUEstimatorSpec=_stub("tf.contrib.tpu.TPUEstimatorSpec"),
    CrossShardOptimizer=_stub("tf.contrib.tpu.CrossShardOptimizer"),
    AsyncCheckpointSaverHook=_stub("tf.contrib.tpu.AsyncCheckpointSaverHook"),
    bfloat16_scope=_stub("tf.contrib.tpu.bfloat16_scope"),
)

opt = types.SimpleNamespace(
    MovingAverageOptimizer=_stub("tf.contrib.opt.MovingAverageOptimizer"),
)

util = types.SimpleNamespace(
    constant_value=_stub("tf.contrib.util.constant_value"),
)
predictor = types.SimpleNamespace(
    from_saved_model=_stub("tf.contrib.predictor.from_saved_model"),
)
graph_editor = types.SimpleNamespace()
seq2seq = types.SimpleNamespace()
