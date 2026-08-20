"""对抗攻击算法集：用于评估各类水印的鲁棒性。"""

from cthulhu_backend.attacks import collusion, geometric, spatial, temporal

__all__ = ["collusion", "geometric", "spatial", "temporal"]
