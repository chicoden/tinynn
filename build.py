import os
import subprocess

CONSOLE_SET_FG_RED = "\033[31m"
CONSOLE_RESET = "\033[0m"

INCLUDE = "include"
SOURCE = "src"
TARGET = "lib"

extra_options = [
    "-O3",
    "-Wall",
    "-Wextra",
    "-Wno-unused-function",
    "-march=native",
    "-ffast-math",
    "-fopt-info-optall-optimized-missed",
    #"-fsanitize=undefined",
    #"-fsanitize=address"
]

source_files = [
    file
    for file in os.listdir(SOURCE)
    if file.endswith(".c")
]

for file in source_files:
    name = file[:file.rfind(".")]
    header_path = os.path.join(INCLUDE, name + ".h")
    src_path = os.path.join(SOURCE, file)
    if not os.path.exists(header_path):
        print(f"{CONSOLE_SET_FG_RED}No header for {src_path}{CONSOLE_RESET}")

for file in os.listdir(TARGET):
    os.remove(os.path.join(TARGET, file))

modules = []
for file in source_files:
    name = file[:file.rfind(".")]
    src_path = os.path.join(SOURCE, file)
    obj_path = os.path.join(TARGET, name + ".o")
    subprocess.call(["gcc", src_path, "-o", obj_path, "-c", *extra_options])
    modules.append(obj_path)

subprocess.call(["gcc", "main.c", "-o", "main.exe", *modules, *extra_options])