"""Historical V11 packer retired; source transplants bypassed canonical respond.

Use stage2.export for future uniquely versioned source/model packages. The
repair PR does not overwrite V11 or any later release, select a trained model,
run a tournament, or publish a package.
"""


def main():
    raise RuntimeError('retired V11 packer: use canonical stage2.export and the current AGENTS.md release contract')


if __name__=='__main__':
    main()
