"""
AceTaskAgent UI Launcher

Starts the web UI server for workflow design, execution, monitoring, and alerts.

Usage:
    python start_ui.py
    python start_ui.py --port 8550
    python start_ui.py --host 0.0.0.0 --port 9000
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="AceTaskAgent Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8550, help="Port to bind (default: 8550)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install it with:")
        print("  pip install uvicorn[standard]")
        sys.exit(1)

    try:
        import fastapi
    except ImportError:
        print("Error: fastapi is required. Install it with:")
        print("  pip install fastapi")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║               AceTaskAgent - Workflow Studio             ║
║                                                          ║
║   Dashboard:    http://{args.host}:{args.port}              ║
║   Designer:     http://{args.host}:{args.port}/#designer    ║
║   Executions:   http://{args.host}:{args.port}/#executions  ║
║   Monitor:      http://{args.host}:{args.port}/#monitor     ║
║   Alerts:       http://{args.host}:{args.port}/#alerts      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "ui.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
