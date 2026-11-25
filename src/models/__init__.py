"""Detector registry. Each owner adds their model here so the shared training
script can build it by name, e.g. `build_model("rgb_resnet50", num_outputs=5)`.

  rgb_resnet50   Person A
  dct_resnet50   Person B
  clip_probe     Person C
"""

MODELS = {}


def register(name):
    def deco(fn):
        MODELS[name] = fn
        return fn
    return deco


def build_model(name, **kwargs):
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODELS)}")
    return MODELS[name](**kwargs)
