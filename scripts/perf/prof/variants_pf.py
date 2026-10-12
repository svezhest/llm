"""Variant setup functions for ab_slab.py."""
import mlx.core as mx


def make(W, pfqmm, lm):
    from mlx_vlm.models.qwen3_5 import gated_delta as gd
    orig_kernel = gd.gated_delta_kernel
    if W._GDN_KERNEL is None:
        W._GDN_KERNEL = orig_kernel

    def base():
        pfqmm.ACTIVE = False
        gd.gated_delta_kernel = orig_kernel

    def pf():
        pfqmm.ACTIVE = True
        gd.gated_delta_kernel = orig_kernel

    def gdnfast():
        pfqmm.ACTIVE = True
        gd.gated_delta_kernel = W._fast_gdn_kernel

    return {"base": base, "pf": pf, "gdnfast": gdnfast}
