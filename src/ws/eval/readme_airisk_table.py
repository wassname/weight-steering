"""Compatibility wrapper for the moved CLI script."""

from ws.scripts.readme_airisk_table import *  # noqa: F401,F403


if __name__ == "__main__":
    from ws.scripts.readme_airisk_table import main

    main()
