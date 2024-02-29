from flash_attn_jax import flash_mha
import functools


def flash_attention_wrapper(q, k, v, attention_width):
    # attention_args = {"q": q, "k": k, "v": v, "window_size": (attention_width, -1), "causal": True}

    return functools.partial(
        flash_mha(q,k,v,softmax_scale=None, is_causal=True, window_size=(attention_width,-1))
    )