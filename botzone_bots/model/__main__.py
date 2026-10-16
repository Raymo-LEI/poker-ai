"""Botzone zip entry. One process, one decision, then exit.

TexasHoldem2p is JSON-only. stdin is the single JSON object documented
on the Bot wiki (``requests``, ``responses``, ...). stdout is one line,
``{"response": <int>}``.

Thread pools are pinned before NumPy is imported. ``decide`` is imported
only after that, and it pins the same variables again before importing
``forward`` (the module that imports NumPy). Botzone gives the process
one core. The rule-bot entry does not import NumPy, so it does not set
these.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys


def main():
    try:
        import decide
        decide.run()
    except Exception:
        sys.stdout.write('{"response": -1}\n')


if __name__ == "__main__":
    main()
