"""Botzone zip entry. One process, one decision, then exit.

TexasHoldem2p is JSON-only. stdin is the single JSON object documented
on the Bot wiki (``requests``, ``responses``, ...). stdout is one line,
``{"response": <int>}``.
"""

import json
import sys


def main():
    try:
        import decide
        decide.run()
    except Exception:
        sys.stdout.write('{"response": -1}\n')


if __name__ == "__main__":
    main()
