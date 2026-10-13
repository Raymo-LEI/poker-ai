"""Read action probabilities from a loaded TF 1.15 NlHoldemNet.

Imported only by the dump and comparison scripts. Those run in the Step-1
Python 3.7 environment. This module does not import torch.
"""

from __future__ import annotations

import numpy as np

from alphanl.nl_obs import masked_softmax_tf


def teacher_distribution(agent, obs):
    """Return ``(probs[5], greedy_action)`` for one wrapper observation.

    Runs the Keras trunk (logits before the mask) inside the policy session,
    then applies the same ``max(log(legal), float32.min)`` mask as
    ``NlHoldemNet.forward``.
    """

    policy = agent._agent.oppo_policy
    base = policy.model.base_model
    sess = policy.get_session()
    arrays = {
        "card_info": np.asarray(obs["card_info"], dtype=np.float32)[None],
        "action_info": np.asarray(obs["action_info"], dtype=np.float32)[None],
        "extra_info": np.asarray(obs["extra_info"], dtype=np.float32)[None],
    }
    feed = {}
    used = set()
    for tensor in list(base.inputs):
        name = str(getattr(tensor, "name", "")).lower()
        matched = None
        for key in arrays:
            if key in name and key not in used:
                matched = key
                break
        if matched is None:
            continue
        feed[tensor] = arrays[matched]
        used.add(matched)
    if set(used) != set(arrays):
        if len(base.inputs) != 3:
            raise RuntimeError("expected 3 keras inputs, got %d" % len(list(base.inputs)))
        feed = {
            base.inputs[0]: arrays["card_info"],
            base.inputs[1]: arrays["action_info"],
            base.inputs[2]: arrays["extra_info"],
        }
    fetched = sess.run(list(base.outputs), feed_dict=feed)
    logits = np.asarray(fetched[0], dtype=np.float64).reshape(-1, 5)[0]
    legal = np.asarray(obs["legal_moves"], dtype=np.float32)
    probs = masked_softmax_tf(logits, legal)[0]
    greedy = int(np.argmax(probs))
    return probs, greedy
