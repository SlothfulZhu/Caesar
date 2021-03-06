"""LCM type definitions
This file automatically generated by exlcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class ResultA(object):
    __slots__ = ["timestamp", "ans"]

    def __init__(self):
        self.timestamp = 0
        self.ans = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(ResultA._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qd", self.timestamp, self.ans))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != ResultA._get_packed_fingerprint():
            raise ValueError("Decode error")
        return ResultA._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = ResultA()
        self.timestamp, self.ans = struct.unpack(">qd", buf.read(16))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if ResultA in parents: return 0
        tmphash = (0x5722ea20a6d952b4) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if ResultA._packed_fingerprint is None:
            ResultA._packed_fingerprint = struct.pack(">Q", ResultA._get_hash_recursive([]))
        return ResultA._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

