from setuptools import find_packages, setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name = "or-pcd",
    version = "0.0.1-BETA.7",
    description = "A python package to perform pointcloud registration with scale differences using a two-block optimization approach",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dpolimeni/Multi-scale-pointcloud-registration",
    author="Diego Polimeni and Alessandro Pannone",
    author_email="a.pannone1798@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy==1.26.4",
        "scikit-learn==1.4.1.post1",
        "open3d==0.18.0"
    ],
    package_data={"": ["data/*.conf", "data/*.ply"]},
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.11",
) 