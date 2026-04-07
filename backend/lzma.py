"""
Mock lzma module to satisfy torchvision import on systems where 
Python was compiled without _lzma support.
"""

class LZMAError(Exception):
    pass

class LZMACompressor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LZMA compression is not supported on this system.")

class LZMADecompressor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LZMA decompression is not supported on this system.")

def is_check_supported(check):
    return False

def open(*args, **kwargs):
    raise NotImplementedError("LZMA open is not supported on this system.")

def compress(data, format=None, check=-1, preset=None, filters=None):
    raise NotImplementedError("LZMA compress is not supported on this system.")

def decompress(data, format=None, memlimit=None, filters=None):
    raise NotImplementedError("LZMA decompress is not supported on this system.")

# Add constants often used
FORMAT_AUTO = 0
FORMAT_XZ = 1
FORMAT_ALONE = 2
FORMAT_RAW = 3
CHECK_NONE = 0
CHECK_CRC32 = 1
CHECK_CRC64 = 4
CHECK_SHA256 = 10
