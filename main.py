"""Development entrypoint."""

import logging

from rich.logging import RichHandler

# NOTE: We only ever run the main entrypoint as main during development.
# Otherwise, main:app is used by guvicorn.


def create_debug_app():
    """Hacky workaround to enable different logging for debug."""
    import fastapi
    import starlette
    import uvicorn

    logging.basicConfig(
        format=None,
        handlers=[
            RichHandler(
                rich_tracebacks=True, tracebacks_suppress=[fastapi, starlette, uvicorn]
            )
        ],
    )
    logging.getLogger("vta_rag").setLevel(logging.DEBUG)

    from vta_rag import app

    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(format=None, handlers=[RichHandler(rich_tracebacks=True)])

    # TODO: Get rich logging working.
    uvicorn.run(
        "main:create_debug_app",
        host="localhost",
        port=8000,
        log_level=logging.INFO,
        reload=True,
        factory=True,
        log_config=None,
    )
