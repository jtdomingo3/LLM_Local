"""
Entry point for running the LLM Local Microservice.

Usage:
    python run.py
    python run.py --port 8100 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="LLM Local Microservice Server")
    parser.add_argument(
        "--host",
        type=str,
        default=settings.service_host,
        help=f"Host interface to bind to (default: {settings.service_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.service_port,
        help=f"Port number to bind to (default: {settings.service_port})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on code change (default: False to save memory)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  🚀 LLM Local Microservice")
    print(f"  • Endpoint:       http://{args.host}:{args.port}")
    print(f"  • Documentation:  http://localhost:{args.port}/docs")
    print(f"  • Health Check:   http://localhost:{args.port}/health/ready")
    print(f"  • Models:         http://localhost:{args.port}/api/models")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
