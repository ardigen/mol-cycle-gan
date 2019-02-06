#!/bin/sh

git submodule update --init --recursive

cat > jtvae/__init__.py << EOF
from .jtnn.mol_tree import Vocab, MolTree
from .jtnn.jtnn_vae import JTNNVAE
EOF