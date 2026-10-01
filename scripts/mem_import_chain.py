"""Print who first imported each heavy package during iris-api import + lifespan startup."""

import asyncio
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import mem_import_profile as mip

HEAVY = sys.argv[1:] or [
    "scipy",
    "sklearn",
    "pandas",
    "pyarrow",
    "numpy",
    "chromadb",
    "dateparser",
    "grpc",
    "pdfminer",
    "lxml",
    "bs4",
    "google",
    "trafilatura",
    "cv2",
    "duckdb",
    "faster_whisper",
    "piper",
    "langchain",
    "langchain_community",
    "phoenix",
    "onnxruntime",
    "tokenizers",
    "torch",
    "joblib",
    "openpyxl",
    "pdfplumber",
    "PIL",
]


def chain(name):
    out = [name]
    while name in mip.PARENT and mip.PARENT[name] != "<entry>":
        name = mip.PARENT[name]
        out.append(name)
        if len(out) > 12:
            break
    return out


async def run():
    sys.meta_path.insert(0, mip._Finder())
    mod = importlib.import_module("iris_harness.server.iris_api.main")
    async with mod.app.router.lifespan_context(mod.app):
        for h in HEAVY:
            # ORDER is completion order; PARENT keys, in insertion order, give the first *start*.
            starts = [n for n in mip.PARENT if n == h or n.startswith(h + ".")]
            if not starts:
                print(f"{h:22s} not imported")
                continue
            c = chain(starts[0])
            kb = sum(v for k, v in mip.SELF.items() if k == h or k.startswith(h + "."))
            print(
                f"{h:22s} {kb/1024:6.1f} MiB  <- "
                + " <- ".join(x for x in c[1:] if not x.startswith(h))
            )


asyncio.run(run())
