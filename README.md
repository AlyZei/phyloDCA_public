# phyloDCA_public
Public repository for 'Towards coevolution-aware ancestral sequence reconstruction': https://www.biorxiv.org/content/10.64898/2026.06.08.731024v1

## Repository structure and one-command download

To make terminal-based download easy, keep files in a simple layout like:

```text
phyloDCA_public/
├── utils/
├── notebooks/
│   └── basic_notebook.ipynb
└── data/
    └── dataset.zip
```

Then users can download everything in one command:

```bash
git clone https://github.com/AlyZei/phyloDCA_public.git
```

If users do not want git history, they can download a single ZIP snapshot:

```bash
curl -fL -o phyloDCA_public.zip https://github.com/AlyZei/phyloDCA_public/archive/refs/heads/main.zip && unzip phyloDCA_public.zip
```
