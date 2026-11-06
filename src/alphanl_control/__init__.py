"""Historical AlphaNLHoldem control group. Not the main model.

The Python package ``alphanl`` keeps its name, but the code that trains
and evaluates this repository lives there: SD-CFR, the betting-history
Transformer, Stage 2, and the Botzone kernel. None of those import this
package unless a caller explicitly selects the ``alphanl_conv`` encoder.

What lives here:

* ``nl_obs`` — NumPy port of upstream ``NlHoldemEnvWrapper._get_observation``
* ``conv_net`` / ``conv_obs`` — PyTorch port of upstream ``NlHoldemNet``
* ``tf_teacher`` / ``inference`` — TensorFlow 1.15 reader and the unchanged
  upstream net, loaded from the ``third_party/AlphaNLHoldem`` submodule
* ``bots`` / ``matchup`` — step-1 random and rule opponents written for that
  upstream environment

Mainline training does not need the submodule or this package.
See ``NOTICE`` at the repository root.
"""

__all__ = [
    "bots",
    "conv_net",
    "conv_obs",
    "inference",
    "matchup",
    "nl_obs",
    "paths",
    "tf_teacher",
]
