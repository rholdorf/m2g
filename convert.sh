/Applications/Blender.app/Contents/MacOS/Blender -b -P ./mixamo_batch.py -- \
   --character ./input/y-bot.fbx \
   --anims ./input/anims/idle.fbx ./input/anims/walking.fbx ./input/anims/running.fbx ./input/anims/punching.fbx \
   --out ./output \
   --mode inplace \
   --export-mesh-in-anims false \
   --apply-scale true \
   --fix-root true \
   --limit-weights 4 \
   --abort-on-incompatible false
