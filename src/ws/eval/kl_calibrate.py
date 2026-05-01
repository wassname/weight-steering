"""Compatibility wrapper for the moved calibration module."""

from ws.kl_calibrate import *  # noqa: F401,F403


if __name__ == "__main__":
    import tyro
    from ws.kl_calibrate import KLCalibrateCfg, main

    main(tyro.cli(KLCalibrateCfg))
