"""PyInstaller 入口：以打包后的 sidecar 方式运行后端服务。"""

import multiprocessing

from cthulhu_backend.main import main

if __name__ == "__main__":
    # 冻结版启用 CTHULHU_PROCESS_WORKERS 时，spawn 子进程会重新执行本入口，
    # 必须先完成多进程引导，否则会递归拉起进程。
    multiprocessing.freeze_support()
    main()
