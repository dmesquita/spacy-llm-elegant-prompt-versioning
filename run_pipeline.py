import os
from pathlib import Path
from typing import Optional

import typer
from wasabi import msg

from spacy_llm.util import assemble
from quotecontextextract import QuoteContextExtractTask

Arg = typer.Argument
Opt = typer.Option

def run_pipeline(
    # fmt: off
    text: str = Arg("", help="Text to perform text categorization on."),
    config_path: Path = Arg(..., help="Path to the configuration file to use."),
    verbose: bool = Opt(False, "--verbose", "-v", help="Show extra information."),
    # fmt: on
):
    if not os.getenv("OPENAI_API_KEY", None):
        msg.fail(
            "OPENAI_API_KEY env variable was not found. "
            "Set it by running 'export OPENAI_API_KEY=...' and try again.",
            exits=1,
        )

    msg.text(f"Loading config from {config_path}", show=verbose)
    nlp = assemble(
        config_path
    )
    doc = nlp(text)

    msg.text(f"Quote: {doc.text}")
    msg.text(f"Context: {doc._.context}")

if __name__ == "__main__":
    typer.run(run_pipeline)
