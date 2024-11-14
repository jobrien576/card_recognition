"""
    Setup file for jo_card_recognition.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.6.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

from setuptools import setup

if __name__ == "__main__":
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            package_dir={"": "src"}  # Ensure src directory is recognized
        )
    except Exception as e:
        print(
            "\n\nAn error occurred while building the project:\n"
            f"{e}\n"
            "Please ensure you have the most updated versions of setuptools, "
            "setuptools_scm, and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
