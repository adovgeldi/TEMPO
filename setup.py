from setuptools import setup, find_packages


# Read the custom license file
with open("LICENSE", "r", encoding="utf-8") as lh:
    license_text = lh.read()

# Read the requirements from requirements.txt
with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="tempo",
    version="0.1.0",
    author="Inspire11",
    description="TEMPO: Time-series Engine for Modeling and Parameter Optimization",
    license="Proprietary - See LICENSE file",
    packages=find_packages(
        where=".",
        include=["tempo", "tempo.*"]
    ),
    package_dir={"": "."},
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11,<3.12",
    extras_require={
        "dev": ["pytest==7.4.3", "ipykernel", "black", "flake8", "pytest-mock"]
    },
    include_package_data=True,
    keywords=["forecasting", "time-series", "machine-learning"],
)
