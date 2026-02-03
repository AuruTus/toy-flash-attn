#!/usr/bin/env python3

import os
import sys

import pathlib

from typing import Union, Optional

# ===============================
#           local modules
# ===============================

SCRIPTS_DIR = pathlib.Path(os.path.dirname(__file__))
sys.path.append(os.path.abspath(SCRIPTS_DIR.parent))

from toy_attn.flash_attn_v2.kernel_configs import (  # noqa: E402
    get_kernels_to_build,
    FlashForwardKernelConfig,
)

INDENTATION = "    "


class TemplateBase:
    def __init__(self, children: Optional[Union[list["TemplateBase"], "TemplateBase"]] = None):
        if children is None:
            self.children = []
        elif isinstance(children, list):
            self.children = children
        else:
            self.children = [children]

    def render(self, depth=0):
        return "\n".join(c.render(depth) for c in self.children)


class KernelConfig(TemplateBase):
    def __init__(self, cfg: FlashForwardKernelConfig):
        super().__init__()
        self.cfg = cfg

    def render(self, depth=0) -> str:
        level_indent = INDENTATION * depth
        cpp_struct_str = self.cfg.to_cpp_struct()
        comment = level_indent + f"// {self.cfg.short_form()}\n"
        map_item = (
            level_indent + f"{{{cpp_struct_str}, &{self.cfg.kernel_name()}<StaticForwardKernelConfig<{cpp_struct_str}>>}}"
        )
        return comment + map_item


class StaticAssertion(TemplateBase):
    def __init__(self, configs: list[FlashForwardKernelConfig]):
        self.configs = configs
        if not self.configs:
            raise ValueError("StaticAssertion: configs should not be empty")

    def render(self, depth=0) -> str:
        level_indent = INDENTATION * depth
        comment = "// explicitly instantiate template and assert concepts to help clangd analyze\n"
        assertions = []
        for c in self.configs:
            assertions.append(f"{level_indent}static_assert(kernel_trait<StaticForwardKernelConfig<{c.to_cpp_struct()}>>);")
        return comment + "\n".join(assertions)


class MapObject(TemplateBase):
    def __init__(self, map_val_name: str, configs: list[KernelConfig]):
        super().__init__(configs)
        self.map_val_name = map_val_name
        if not self.children:
            raise ValueError("MapObject: configs should not be empty")

    def render(self, depth=0) -> str:
        level_indent = INDENTATION * depth
        definition = f"{level_indent}auto forward_kernels = std::map<FlashForwardKernelConfig, {self.map_val_name}>"
        scope_beg = "{\n"
        scope_end = "\n};"
        configs = []
        for c in self.children:
            configs.append(c.render(depth + 2))
        body = ",\n".join(configs)
        return definition + scope_beg + body + scope_end


class UsingDecl(TemplateBase):
    def __init__(self, name: str, type_expr: str):
        super().__init__()
        self.name = name
        self.type_expr = type_expr

    def render(self, depth=0) -> str:
        return f"{INDENTATION * depth}using {self.name} = {self.type_expr};"


class Namespace(TemplateBase):
    def __init__(self, name: str, children: list[TemplateBase]):
        super().__init__(children)
        self.name = name

    def render(self, depth=0) -> str:
        level_indent = INDENTATION * depth
        header = f"{level_indent}namespace {self.name} {{\n"
        body = "\n\n\n".join(c.render(depth) for c in self.children)
        footer = f"\n{level_indent}}} // namespace {self.name}"
        return header + "\n" + body + footer


class Headers(TemplateBase):
    def __init__(self, includes: list[str]):
        super().__init__()
        self.includes = includes

    def render(self, depth=0) -> str:
        lines = []
        for inc in self.includes:
            if not inc:
                lines.append("")
            elif inc.startswith("<") or inc.startswith('"'):
                lines.append(f"#include {inc}")
            else:
                lines.append(f'#include "{inc}"')
        return "\n".join(lines)


class FileTemplate(TemplateBase):
    def __init__(self, comment: str, headers: Headers, namespace: Namespace):
        super().__init__()
        self.comment = comment
        self.headers = headers
        self.namespace = namespace

    def render(self, depth=0) -> str:
        parts = [
            self.comment,
            "",
            "#pragma once",
            "",
            self.headers.render(depth),
            "",
            self.namespace.render(depth),
        ]
        return "\n".join(parts)


def main():
    raw_kernel_configs = get_kernels_to_build()
    kernel_configs = [KernelConfig(cfg) for cfg in raw_kernel_configs]
    for cfg in raw_kernel_configs:
        print(cfg.short_form())

    map_val_name = "forward_kernel_fn"
    file_template = FileTemplate(
        comment='// This file is auto-generated in "gen_kernel_instantiations.py".',
        headers=Headers(
            [
                "<map>",
                "",
                '"concepts.h"',
                '"flash_attention.cuh"',
                '"forward_kernel.cuh"',
            ]
        ),
        namespace=Namespace(
            "flash_attn_v2",
            [
                UsingDecl(map_val_name, "void (*)(const ForwardKernelArgs)"),
                StaticAssertion(raw_kernel_configs),
                MapObject(map_val_name, kernel_configs),
            ],
        ),
    )

    output = file_template.render()

    with open("./csrc/flash_attn_v2/include/flash_kernels.cuh", "w") as f:
        f.write(output)
        f.write("\n")


if __name__ == "__main__":
    main()
