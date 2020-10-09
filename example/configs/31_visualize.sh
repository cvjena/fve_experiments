
CLASS_NAMES=${CLASS_NAMES:-"${HOME}/Data/DATASETS/birds/cub200/classes.txt"}
CLASSES=${CLASSES:-"1 9 53 67 81 115 137 172 185 198"}

# Albatrosses (3 classes)
# CLASSES="$(seq 1 3)"

# Blackbirds (4 classes)
CLASSES="$(seq 9 12)"

# Flycatchers (7 classes)
# CLASSES="$(seq 37 43)"

# Woodpeckers (6 classes)
# CLASSES="$(seq 187 192)"

# Warblers (25 classes)
# CLASSES="$(seq 158 182)"

# Wrens (7 classes)
# CLASSES="$(seq 193 199)"


OPTS="${OPTS} --classes ${CLASSES}"
OPTS="${OPTS} --class_names ${CLASS_NAMES}"
