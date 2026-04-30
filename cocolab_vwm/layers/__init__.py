"""Single-layer and hierarchical OCOS Nengo networks."""
from cocolab_vwm.layers.hierarchy import make_hierarchy
from cocolab_vwm.layers.nengo_layer import make_ocos_layer
from cocolab_vwm.layers.pooled_hierarchy import make_pooled_hierarchy

__all__ = ["make_ocos_layer", "make_hierarchy", "make_pooled_hierarchy"]
