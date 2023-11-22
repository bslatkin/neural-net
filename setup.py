from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="cneural_net",
            sources=["cneural_net.c"],
        ),
    ]
)
