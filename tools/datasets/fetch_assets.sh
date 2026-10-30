#!/usr/bin/env bash
# Downloads the models used by make_dataset.py, pinned to fixed versions.
#   head:   Lee Perry-Smith head scan, Infinite-Realities, CC BY 3.0 (three.js examples r170)
#   body:   "Michelle", Adobe Mixamo (three.js examples r170) - internal testing only
#   lizard: hornedlizard.spz, Niantic, MIT (nianticlabs/spz)
set -euo pipefail
dest="${1:-assets}"
mkdir -p "$dest"
three=https://raw.githubusercontent.com/mrdoob/three.js/r170/examples/models/gltf
spz=https://raw.githubusercontent.com/nianticlabs/spz/affd0ecea7fbb4c265ee119475af7ee5b2997482/samples
fetch() { curl -fsSL --retry 3 -o "$dest/$(basename "$1")" "$1"; }
fetch "$three/LeePerrySmith/LeePerrySmith.glb"
fetch "$three/LeePerrySmith/Map-COL.jpg"
fetch "$three/Michelle.glb"
fetch "$spz/hornedlizard.spz"
ls -l "$dest"
