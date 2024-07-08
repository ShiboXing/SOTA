# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件提供了命令行工具的入口逻辑。

Authors: xingshibo(xingshibo@baidu.com)
Date:    2024/07/08 14:30:47
"""

__all__ = [
    'main',
]


def main(args=None):
    """主程序入口"""
    from . import demo
    if args is None:
        # 如果未传入命令行参数，则直接从sys中读取，并过滤掉第0位的入口文件名
        import sys
        args = sys.argv[1:]

    hello = demo.Hello()
    return hello.run(*args)

