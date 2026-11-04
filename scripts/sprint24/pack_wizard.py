"""Retired ZIP-transplant packer. Canonical exports own the complete runtime.

Use alphanl.stage2.export.pack_botzone_zip(path, release_id='Vn_...') for a
future versioned package. Allocate an unused increasing Vn, bind source and
model metadata, and satisfy AGENTS.md hashes/debug/replay verification. This
repair PR intentionally creates no upload package or production model.
"""

def prepare_tree(*args,**kwargs):
    raise RuntimeError('retired source transplant: use canonical stage2.export; allocate a fresh Vn and follow AGENTS.md')


def write_model(*args,**kwargs):
    raise RuntimeError('legacy Vleaf/CFV attachment is incompatible with combo_value_v1; generate new labels and train first')


def pack(*args,**kwargs):
    raise RuntimeError('historical V10 packer is retired; no implicit release, upload or deployment')


if __name__=='__main__':
    pack()
