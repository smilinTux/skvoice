"""Entry point for python -m skvoice."""

import uvicorn

from skvoice.config import Config


def main():
    uvicorn.run(
        "skvoice.service:app",
        host=Config.HOST,
        port=Config.PORT,
    )


if __name__ == "__main__":
    main()
