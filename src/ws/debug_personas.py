"""Compatibility wrapper for the moved CLI script."""

from ws.scripts.debug_personas import *  # noqa: F401,F403


if __name__ == "__main__":
    import tyro
    from ws.scripts.debug_personas import PersonaDebugCfg, main

    main(tyro.cli(PersonaDebugCfg))
