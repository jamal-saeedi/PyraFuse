from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nap-26")
except PackageNotFoundError:
    __version__ = "dev"
