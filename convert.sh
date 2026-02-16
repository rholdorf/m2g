## com inplace
#/Applications/Blender.app/Contents/MacOS/Blender -b -P ./mixamo_batch.py -- \
#   --character ./input/x-bot.fbx \
#   --anims \
#   --out ./output \
#   --mode inplace \
#   --inplace-ground-lock true \
#   --export-mesh-in-anims false \
#   --apply-scale true \
#   --fix-root true \
#   --limit-weights 4 \
#   --abort-on-incompatible false

## Sem inplace
/Applications/Blender.app/Contents/MacOS/Blender -b -P ./mixamo_batch.py -- \
   --character ./input/x-bot.fbx \
   --anims \
   ./input/anims/idle.fbx \
   ./input/anims/dying.fbx \
   ./input/anims/look_around.fbx \
   ./input/anims/ninja_idle.fbx \
   ./input/anims/offensive_idle.fbx \
   ./input/anims/punching.fbx \
   --out ./output \
   --export-mesh-in-anims false \
   --apply-scale true \
   --fix-root true \
   --limit-weights 4 \
   --abort-on-incompatible false

## com inplace anchor foot
/Applications/Blender.app/Contents/MacOS/Blender -b -P ./mixamo_batch.py -- \
   --character ./input/x-bot.fbx \
   --anims \
   ./input/anims/right_strafe.fbx \
   ./input/anims/right_strafe_walk.fbx \
   ./input/anims/jumping.fbx \
   ./input/anims/left_strafe_walk.fbx \
   ./input/anims/left_strafe.fbx \
   ./input/anims/catwalk_walk_forward.fbx \
   ./input/anims/catwalk_walking.fbx \
   ./input/anims/walking.fbx \
   ./input/anims/running.fbx \
   --out ./output \
   --mode inplace \
   --inplace-ground-lock true \
   --inplace-anchor foot \
   --export-mesh-in-anims false \
   --apply-scale true \
   --fix-root true \
   --limit-weights 4 \
   --abort-on-incompatible false
