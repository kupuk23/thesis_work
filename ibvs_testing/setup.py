from setuptools import find_packages, setup
from glob import glob
import os

package_name = "ibvs_testing"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name, "scripts"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tafarrel",
    maintainer_email="kupuk23@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["ibvs_node = scripts.ibvs_node:main"],
    },
)
