"""Botzone zip entry: pin BLAS threads before NumPy, then decide once and exit."""

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import sys


def _main():
    try:
        import botzone_entry
    except Exception:
        sys.stdout.write('{"response":-1}\n')
        return
    botzone_entry.main()


if __name__ == "__main__":
    _main()
