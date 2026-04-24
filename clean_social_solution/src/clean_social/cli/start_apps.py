from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from pathlib import Path

from clean_social.utils.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start both FastAPI and Streamlit for CleanSocial.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument(
        "--keep-running-on-exit",
        action="store_true",
        default=False,
        help="Do not terminate child processes when launcher exits.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        default=False,
        help="Start services in detached mode and exit immediately.",
    )
    return parser.parse_args()


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _popen_kwargs(detach: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {"text": True}
    if detach:
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        kwargs["stdin"] = subprocess.DEVNULL
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return kwargs


def launch_api(host: str, port: int, root: Path, detach: bool) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "clean_social.apps.api:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(command, cwd=str(root), **_popen_kwargs(detach))


def launch_streamlit(port: int, root: Path, detach: bool) -> subprocess.Popen[str]:
    app_path = root / "src" / "clean_social" / "apps" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    return subprocess.Popen(command, cwd=str(root), **_popen_kwargs(detach))


def wait_for_port(host: str, port: int, timeout_seconds: float = 20.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def wait_for_api_health(host: str, port: int, timeout_seconds: float = 20.0) -> bool:
    deadline = time.time() + timeout_seconds
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def main() -> None:
    args = parse_args()
    root = project_root()

    children: list[subprocess.Popen[str]] = []

    api_url = f"http://{args.host}:{args.api_port}"
    ui_url = f"http://{args.host}:{args.ui_port}"

    api_running = is_port_open(args.host, args.api_port)
    ui_running = is_port_open(args.host, args.ui_port)

    if api_running:
        if wait_for_api_health(args.host, args.api_port, timeout_seconds=3.0):
            print(f"API already running on {api_url}")
        else:
            raise RuntimeError(
                f"Port {args.api_port} is occupied but {api_url}/health is not responding. "
                "Free this port or choose --api-port."
            )
    else:
        print(f"Starting API on {api_url}")
        children.append(launch_api(args.host, args.api_port, root, args.detach))

    if ui_running:
        print(f"Streamlit already running on {ui_url}")
    else:
        print(f"Starting Streamlit on {ui_url}")
        children.append(launch_streamlit(args.ui_port, root, args.detach))

    for proc in children:
        if proc.poll() is not None:
            raise RuntimeError("One of the app processes exited immediately. Check terminal output for details.")

    if not api_running:
        if not wait_for_port(args.host, args.api_port, timeout_seconds=20.0):
            raise RuntimeError(f"API did not bind to {api_url} in time.")
        if not wait_for_api_health(args.host, args.api_port, timeout_seconds=20.0):
            raise RuntimeError(f"API health check did not become ready at {api_url}/health.")

    if not ui_running:
        if not wait_for_port(args.host, args.ui_port, timeout_seconds=20.0):
            raise RuntimeError(f"Streamlit did not bind to {ui_url} in time.")

    print(f"\nAPI URL: {api_url}")
    print(f"Streamlit URL: {ui_url}")

    if not children:
        print("Both services were already active. Nothing new to launch.")
        return

    if args.detach:
        print("Services started in detached mode.")
        return

    print("Press Ctrl+C to stop services started by this launcher.")
    try:
        while True:
            for proc in children:
                if proc.poll() is not None:
                    raise RuntimeError("One of the app processes stopped unexpectedly.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping launched services...")
    finally:
        if not args.keep_running_on_exit:
            for proc in children:
                if proc.poll() is None:
                    proc.terminate()
            for proc in children:
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        proc.kill()


if __name__ == "__main__":
    main()
