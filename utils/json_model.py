import json
import torch.nn as nn
import model.layer as L  # noqa:F401


def load_model_from_json(f: str) -> dict[str, nn.Sequential]:
    """
    JSON format
    -----------
    [
        "dependency (optional)": [
            {
                "module": dependency1_class,
                "as": alias
            },
            ...
        ],
        "model_name1": [
            {
                "class": class_name,
                "args": [val, ...],
                "kwargs": {"key": val, ...}
            },
            ... (layers)
        ],
        "model_name2": { ... }
    ]
    """
    with open(f) as fp:
        data = json.load(fp)

    # resolve dependency
    dep_list = data.pop("dependency", None)
    if dep_list:
        for dep in dep_list:
            eval(f"import {dep['module']} as {dep['as']}")

    models = {}
    for name, layers in data.items():
        tmp = nn.Sequential()
        for i, layer in enumerate(layers):
            args_str = [f"[{','.join(map(str, arg))}]" if isinstance(arg, list)
                        else str(arg)
                        for arg in layer["args"]]
            kwargs_str = [f"{k}=[{','.join(v)}]" if isinstance(v, list)
                          else "{!s}={!s}".format(k, v)
                          for k, v in layer["kwargs"].items()]
            instance = eval(
                f"{layer['class']}({','.join(args_str + kwargs_str)})")
            tmp.add_module(f"{name}/layer{i}", instance)
        models[name] = tmp
    return models
