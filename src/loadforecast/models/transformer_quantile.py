"""Seq2seq Transformer with quantile output heads (P10 / P50 / P90).

Drop-in architectural alternative to `build_lstm_quantile`: identical I/O
contract (inputs [encoder_in (672, enc_features), decoder_in (96, dec_features)],
output (96, 3)). Trained with the SAME pinball loss and `compile_lstm_quantile`
from lstm_quantile.py — only the body between the inputs and the
TimeDistributed(Dense(3)) head changes. This is a comparison-baseline
experiment, not a production model.

Positional encoding is a tiny *registered serializable* layer holding a
NON-trainable sinusoidal table. That choice matters: `models/predict.py`
calls `keras.models.load_model(..., compile=False)` with NO custom_objects.
A Lambda-based positional add would be blocked by Keras 3 safe_mode on load;
a registered Layer reconstructs from the global registry instead. Any process
that loads a transformer checkpoint must therefore import THIS module first so
the `@register_keras_serializable` decorator fires — the training scripts do
(they call `build_transformer_quantile`) and the compare scripts add an explicit
`import loadforecast.models.transformer_quantile`. We deliberately do NOT import
it from `models/__init__.py`, which would force an eager TensorFlow import on
every `loadforecast.models.*` import and slow the TF-free XGBoost API path.
Non-trainable PE => zero extra trainable params, which matters at this data size.
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from .dataset import DEC_FEATURE_NAMES, ENC_FEATURE_NAMES, LOOKBACK_QH, QH_PER_DAY
from .lstm_quantile import QUANTILES


@keras.utils.register_keras_serializable(package="loadforecast")
class SinusoidalPositionalEncoding(layers.Layer):
    """Add fixed sinusoidal positional encodings to (batch, seq, d_model)."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)

    def build(self, input_shape):
        seq_len = int(input_shape[1])
        pos = np.arange(seq_len)[:, None]
        i = np.arange(self.d_model)[None, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(self.d_model))
        angles = pos * angle_rates
        pe = np.zeros((seq_len, self.d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        # Non-trainable weight => round-trips through model.save / load_model.
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(seq_len, self.d_model),
            initializer=keras.initializers.Constant(pe),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_encoding[None, ...]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


def _encoder_block(x, *, d_model, num_heads, ff_dim, dropout, name):
    """Post-norm transformer encoder block (stable for 1-2 layers)."""
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout, name=f"{name}_selfattn",
    )(x, x)  # self-attention, no mask
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(
        layers.Add(name=f"{name}_add1")([x, attn]))
    ff = layers.Dense(ff_dim, activation="relu", name=f"{name}_ff1")(x)
    ff = layers.Dropout(dropout, name=f"{name}_ffdrop")(ff)
    ff = layers.Dense(d_model, name=f"{name}_ff2")(ff)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(
        layers.Add(name=f"{name}_add2")([x, ff]))
    return x


def _decoder_block(x, memory, *, d_model, num_heads, ff_dim, dropout, name):
    """Decoder block: self-attn (no causal mask) + cross-attn to encoder memory + FFN.

    No causal mask: every decoder input is a future-KNOWN covariate (calendar,
    TSO load forecast, weather), not an autoregressive target. There is no
    target leakage, and bidirectional attention over the whole delivery day
    lets each quarter-hour see the full daily shape in one shot.
    """
    sa = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout, name=f"{name}_selfattn",
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(
        layers.Add(name=f"{name}_add1")([x, sa]))
    ca = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout, name=f"{name}_crossattn",
    )(x, memory)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(
        layers.Add(name=f"{name}_add2")([x, ca]))
    ff = layers.Dense(ff_dim, activation="relu", name=f"{name}_ff1")(x)
    ff = layers.Dropout(dropout, name=f"{name}_ffdrop")(ff)
    ff = layers.Dense(d_model, name=f"{name}_ff2")(ff)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln3")(
        layers.Add(name=f"{name}_add3")([x, ff]))
    return x


def build_transformer_quantile(
    *,
    enc_features: int = len(ENC_FEATURE_NAMES),
    dec_features: int = len(DEC_FEATURE_NAMES),
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.1,
    n_quantiles: int = len(QUANTILES),
) -> keras.Model:
    """Seq2seq Transformer with the SAME I/O contract as build_lstm_quantile.

    inputs : [encoder_in (672, enc_features), decoder_in (96, dec_features)]
    output : (96, n_quantiles) via TimeDistributed(Dense)

    Train with num_blocks=1, ff_dim=64 for ~LSTM parameter parity (~45k).
    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # Encoder: project -> +posenc -> N self-attention blocks.
    enc_in = keras.Input(shape=(LOOKBACK_QH, enc_features), name="encoder_in")
    e = layers.Dense(d_model, name="enc_proj")(enc_in)
    e = SinusoidalPositionalEncoding(d_model, name="enc_posenc")(e)
    e = layers.Dropout(dropout, name="enc_indrop")(e)
    for b in range(num_blocks):
        e = _encoder_block(e, d_model=d_model, num_heads=num_heads,
                           ff_dim=ff_dim, dropout=dropout, name=f"enc{b}")
    memory = e  # (batch, 672, d_model)

    # Decoder: project -> +posenc -> N (self + cross + FFN) blocks.
    dec_in = keras.Input(shape=(QH_PER_DAY, dec_features), name="decoder_in")
    d = layers.Dense(d_model, name="dec_proj")(dec_in)
    d = SinusoidalPositionalEncoding(d_model, name="dec_posenc")(d)
    d = layers.Dropout(dropout, name="dec_indrop")(d)
    for b in range(num_blocks):
        d = _decoder_block(d, memory, d_model=d_model, num_heads=num_heads,
                           ff_dim=ff_dim, dropout=dropout, name=f"dec{b}")

    # Quantile head: identical to the LSTM model.
    y = layers.TimeDistributed(
        layers.Dense(n_quantiles, name="quantile_dense"), name="prediction",
    )(d)
    return keras.Model(inputs=[enc_in, dec_in], outputs=y, name="transformer_quantile")


__all__ = ["SinusoidalPositionalEncoding", "build_transformer_quantile"]


if __name__ == "__main__":
    from .lstm_quantile import compile_lstm_quantile
    m = compile_lstm_quantile(build_transformer_quantile(num_blocks=1, ff_dim=64))
    m.summary(line_length=100)
