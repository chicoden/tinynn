import os
import subprocess

SRC = "src"
LIB = "lib"

extra_options = [
    "-O3",
    "-Wall",
    "-Wextra",
    "-Wno-unused-function",
    "-march=native",
    "-ffast-math",
    "-fopt-info-optall-optimized-missed"
]

modules = []
for file in os.listdir(SRC):
    name, extension = file.rsplit(".", maxsplit=1)
    if extension == "c":
        src_path = os.path.join(SRC, file)
        obj_path = os.path.join(LIB, name + ".o")
        subprocess.call(["gcc", src_path, "-o", obj_path, "-c", *extra_options])
        modules.append(obj_path)

subprocess.call(["gcc", "main.c", "-o", "main.exe", *modules, *extra_options])
