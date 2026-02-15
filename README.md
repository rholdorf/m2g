# m2g

Mixamo FBX to GLB conversion

Script Blender 5.0.1 (Python) para rodar headless e gerar:

- character.glb (mesh + skeleton, sem animações)
- Walk.glb, Run.glb, … (1 GLB por animação; por padrão exporta só skeleton+animação, sem mesh)
- report.json (divergências, correções e validações)

Ele assume “Mixamo padrão” (ex.: bone Hips) e tenta detectar/corrigir divergências comuns (escala, criação de Root, pesos limitados a 4, etc.). Se a animação tiver hierarquia diferente (não mapeável por nome), ele marca como incompatível e segue (ou aborta se você pedir).

```bash
blender -b -P mixamo_batch.py -- \
  --character ./input/Dude.fbx \
  --anims ./input/anims/Walk.fbx ./input/anims/Run.fbx ./input/anims/Idle.fbx \
  --out ./output \
  --mode inplace \
  --export-character true \
  --export-mesh-in-anims false \
  --apply-scale true \
  --fix-root true \
  --limit-weights 4 \
  --abort-on-incompatible false
```

- --mode inplace remove deslocamento horizontal do Hips (in-place).
- --mode rootmotion preserva deslocamento (root motion).
