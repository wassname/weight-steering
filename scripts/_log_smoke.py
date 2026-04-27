"""Smoke test for ws._log token-efficient logging helpers."""
from loguru import logger

from ws._log import final_summary, get_argv, setup_logging


def main() -> None:
    p = setup_logging("test_smoke")
    logger.info("hello plain stdout")
    logger.debug("hello debug-only-in-file")
    final_summary(
        out="out/test.csv",
        argv=get_argv(),
        main_metric="spread=+1.234 pmass_min=0.987",
        cue="🟢",
        table_rows=[[
            "+1.234", "0.987", "sycophancy", "lora", "Qwen3-0.6B",
            "flag=smoke", "out/test.csv",
        ]],
        headers=["spread", "pmass", "behavior", "adapter", "model", "flags", "out"],
        floatfmt="",
    )
    print("VERBOSE LOG PATH:", p)
    print("--- verbose log content ---")
    print(open(p).read())


if __name__ == "__main__":
    main()
