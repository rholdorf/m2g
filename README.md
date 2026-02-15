# m2g

Pipeline offline para converter Mixamo FBX em GLB separado por animação.

Entrada:

- 1 FBX de personagem com skin (`--character`)
- N FBXs de animação (`--anims`)

Saída:

- `<character>.glb`: mesh + skeleton, sem animações
- `<anim>.glb`: skeleton + 1 clip (mesh opcional)
- `report.json`: validações, correções e métricas de bake/export

## Requisitos

- macOS
- Blender 5.x instalado (testado com 5.0.1)
- Arquivos Mixamo com nomes de bones compatíveis (mesmo personagem base)

Executável Blender usado no projeto:

```bash
/Applications/Blender.app/Contents/MacOS/Blender
```

## Uso rápido

### Opção 1: script pronto

```bash
./convert.sh
```

### Opção 2: comando direto

```bash
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
```

## Parâmetros

- `--character <path>`: FBX do personagem com skin
- `--anims <paths...>`: lista de FBXs de animação
- `--out <dir>`: diretório de saída (default `./output`)
- `--mode inplace|rootmotion`:
  - `inplace`: remove deslocamento no mundo (Hips/Root/Object em `location = 0`)
  - `rootmotion`: preserva deslocamento original
- `--export-mesh-in-anims true|false`: inclui mesh nos GLBs de animação (default `false`)
- `--apply-scale true|false`: aplica transform para estabilidade (default `true`)
- `--fix-root true|false`: cria bone `Root` acima de `Hips` quando ausente (default `true`)
- `--limit-weights <N>`: limita influências por vértice (default `4`)
- `--abort-on-incompatible true|false`: aborta ao encontrar animação incompatível (default `false`)

## O que o script faz

1. Importa o personagem FBX.
2. Normaliza nomes de bone (remove prefixo `mixamorig:`).
3. Opcionalmente aplica transform, cria `Root`, limita pesos.
4. Exporta o GLB do personagem (sem animação).
5. Para cada animação FBX:
   - importa animação
   - normaliza nomes de bone
   - valida compatibilidade por conjunto de bones (`common_ratio >= 0.98`)
   - copia F-Curves de pose para uma nova Action no rig base (compatível com API legada e slotted do Blender 5)
   - aplica `inplace` ou preserva root motion
   - exporta GLB com exatamente 1 clip (`export_animation_mode=ACTIVE_ACTIONS`)

## Estrutura esperada

```text
.
├── mixamo_batch.py
├── convert.sh
├── input
│   ├── y-bot.fbx
│   └── anims
│       ├── idle.fbx
│       ├── walking.fbx
│       ├── running.fbx
│       └── punching.fbx
└── output
    ├── y-bot.glb
    ├── idle.glb
    ├── walking.glb
    ├── running.glb
    ├── punching.glb
    └── report.json
```

## report.json (campos principais)

Em `character`:

- `bone_rename`
- `root.created`
- `weights.limited_on_meshes`

Em cada item de `anims`:

- `compat`: compatibilidade entre rig base e rig da animação
- `bake`:
  - `frame_start`, `frame_end`
  - `bones_baked`
  - `curves_copied`
  - `curves_skipped_non_bone`
- `inplace.status`
- `export.path`

## Troubleshooting

- `keyword "export_colors" unrecognized`
  - Normal em algumas versões do Blender 5.
  - O script já remove automaticamente parâmetros não suportados no export glTF.

- `WARNING: User property type 'Short' is not supported`
  - Aviso comum no import FBX.
  - Em geral não impede a conversão.

- GLB de animação com vários clips
  - Corrigido no pipeline atual com `ACTIVE_ACTIONS` + limpeza de actions temporárias.

- Esqueleto deslocando no mundo em `inplace`
  - No estado atual, `inplace` é estrito e zera `location` de Hips/Root/Object para evitar drift.
  - Use `--mode rootmotion` se quiser preservar deslocamento.

- Animação incompatível / distorcida
  - Verifique `report.json -> anims[*].compat`.
  - Regra prática: usar animações geradas para o mesmo personagem Mixamo.
