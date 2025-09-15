# docs_gen/gen_api.py
import pathlib
import pkgutil

import mkdocs_gen_files

PACKAGE = "solshade"


def iter_modules(package: str):
    """Yield fully-qualified module names under `package` (non-packages only)."""
    import importlib

    mod = importlib.import_module(package)
    root = pathlib.Path(str(mod.__file__)).parent
    for module in pkgutil.walk_packages([str(root)], prefix=f"{package}."):
        if not module.ispkg:
            yield module.name


# We will write generated pages under docs/api/
api_dir = pathlib.Path("api")
nav = mkdocs_gen_files.Nav()

for mod_name in sorted(iter_modules(PACKAGE)):
    # Write each module page: docs/api/<package.module>.md
    doc_path = api_dir / f"{mod_name}.md"
    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(f"# `{mod_name}`\n\n")
        f.write(f"::: {mod_name}\n")

    # IMPORTANT: When building the *index* page INSIDE docs/api/,
    # links must be relative to that folder. So use just the filename,
    nav[mod_name] = doc_path.name

# Build an API index page inside docs/api/ that lists the modules
index_path = api_dir / "index.md"
with mkdocs_gen_files.open(index_path, "w") as f:
    f.write("# API Reference\n\n")
    f.writelines(nav.build_literate_nav())
