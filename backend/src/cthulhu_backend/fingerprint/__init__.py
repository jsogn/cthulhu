"""平台去重指纹的本地代理实现。

真实平台（抖音/巨量/千川）的判重细节不公开，这里用与其候选机制同构的
开源可复现代理：感知哈希、DCT 结构签名、浅层特征与音频谱哈希。代理只能
作研究裁判，不能等价于平台判定。
"""

from cthulhu_backend.fingerprint import audiofp, hashes, proxy

__all__ = ["audiofp", "hashes", "proxy"]
