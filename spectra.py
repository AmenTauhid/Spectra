#!/usr/bin/env python3
"""Convenience entry point: python spectra.py <args>"""
import runpy
runpy.run_module("src", run_name="__main__", alter_sys=True)
