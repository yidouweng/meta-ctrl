from setuptools import find_packages, setup

setup(
    name="meta-ctrl",
    version="1.0.0",
    description="Meta-Ctrl: two-level constrained decoding for LLM planners (EAI experiments)",
    python_requires=">=3.11",
    packages=find_packages(include=["ctrlg", "ctrlg.*"]),
)
