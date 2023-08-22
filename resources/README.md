## Not with my name! Inferring artists' names of input strings employed by Diffusion Models - Resources

After the [Installation process](/README.md#installation), the folder structure should be the following:

```
not-with-my-name
├── ...
├── resources
│   ├── ckpts
│   │   │   └── siamese_not_w_my_name.ckpt
│   │   │
│   ├── small-dataset
│   │   │   ├── alfred_sisley
│   │   │   │   ├── ai_generated
│   │   │   │   │   ├── [...].png
│   │   │   │   │   └── [...].png
│   │   │   │   └── original_paintings
│   │   │   │       ├── [...].jpg
│   │   │   │       └── [...].jpg
│   │   │   ├── claude_monet
│   │   │   │   ├── ai_generated
│   │   │   │   │   ├── [...].png
│   │   │   │   │   └── [...].png
│   │   │   │   └── original_paintings
│   │   │   │       ├── [...].jpg
│   │   │   │       └── [...].jpg
│   │   │   ├── pablo_picasso
│   │   │   │   ├── ai_generated
│   │   │   │   │   ├── [...].png
│   │   │   │   │   └── [...].png
│   │   │   │   └── original_paintings
│   │   │   │       ├── [...].jpg
│   │   │   │       └── [...].jpg
│   │   │   ├── paul_cezanne
│   │   │   │   ├── ai_generated
│   │   │   │   │   ├── [...].png
│   │   │   │   │   └── [...].png
│   │   │   │   └── original_paintings
│   │   │   │       ├── [...].jpg
│   │   │   │       └── [...].jpg
│   │   │   └── pierre_auguste_renoir
│   │   │       ├── ai_generated
│   │   │       │   ├── [...].png
│   │   │       │   └── [...].png
│   │   │       └── original_paintings
│   │   │           ├── [...].jpg
│   │   │           └── [...].jpg
│   │   │
│   └── medium-dataset
│           ├── alfred_sisley
│           │   ├── ai_generated
│           │   │   ├── [...].png
│           │   │   └── [...].png
│           │   └── original_paintings
│           │       ├── [...].jpg
│           │       └── [...].jpg
│           ├── claude_monet
│           │   ├── ai_generated
│           │   │   ├── [...].png
│           │   │   └── [...].png
│           │   └── original_paintings
│           │       ├── [...].jpg
│           │       └── [...].jpg
│           ├── pablo_picasso
│           │   ├── ai_generated
│           │   │   ├── [...].png
│           │   │   └── [...].png
│           │   └── original_paintings
│           │       ├── [...].jpg
│           │       └── [...].jpg
│           ├── paul_cezanne
│           │   ├── ai_generated
│           │   │   ├── [...].png
│           │   │   └── [...].png
│           │   └── original_paintings
│           │       ├── [...].jpg
│           │       └── [...].jpg
│           └── pierre_auguste_renoir
│               ├── ai_generated
│               │   ├── [...].png
│               │   └── [...].png
│               └── original_paintings
│                   ├── [...].jpg
│                   └── [...].jpg
│
├── ...