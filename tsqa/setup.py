from setuptools import find_packages, setup

setup(
    name="tsqa",
    version="0.1.0",
    description="T-SMART: structured multimodal reasoning for time-series question answering",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["tsqa", "tsqa.*"]),
    package_data={"tsqa": ["verifier/*.json"]},
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "statsmodels>=0.14",
        "ruptures>=1.1",
        "tqdm>=4.65",
        "google-genai>=1.0",
        # tsqa.eval.runner imports tsqa.tools.cwt at module scope, and cwt.py
        # imports matplotlib at module scope: without this, importing the
        # pipeline entry point fails outright.
        "matplotlib>=3.7",
        # tsqa.llm.client decodes rendered artifacts for the vision sensor.
        "pillow>=10.0",
    ],
    extras_require={
        # Alternate backbones. The paper's results all use google-genai.
        "openai": ["openai>=1.0"],
        "qwen": ["torch>=2.0", "transformers>=4.40"],
        # Benchmark loading (TimeSeriesExam / MMTS-Bench harnesses).
        "bench": ["datasets>=2.14", "pandas>=2.0", "python-dotenv>=1.0"],
        "dev": ["pytest>=7.0"],
    },
)
