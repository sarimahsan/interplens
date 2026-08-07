"""CLI entrypoint for InterpLens."""

import argparse
import sys
from .config import settings
from .utils.device import resolve_device, get_vram_usage


def main():
    parser = argparse.ArgumentParser(description="InterpLens Mechanistic Interpretability Debugger")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")
    
    # launch command
    launch_parser = subparsers.add_parser("launch", help="Launch InterpLens debugger Web UI")
    launch_parser.add_argument("--model", type=str, default="gpt2", help="Model name or path (default: gpt2)")
    launch_parser.add_argument("--host", type=str, default=settings.host, help="Host address (default: 127.0.0.1)")
    launch_parser.add_argument("--port", type=int, default=settings.port, help="Port (default: 8501)")
    launch_parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, mps, auto)")
    launch_parser.add_argument("--hf-token", "--token", type=str, default=None, help="HuggingFace access token for gated models")
    
    args = parser.parse_args()
    
    if args.command == "launch":
        device = resolve_device(getattr(args, "device", "auto"))
        host = getattr(args, "host", settings.host)
        port = getattr(args, "port", settings.port)
        model_name = getattr(args, "model", "gpt2")
        hf_token = getattr(args, "hf_token", None)
        
        print(f"🚀 Starting InterpLens Debugger v{settings.version}")
        print(f"📍 Model: {model_name} | Device: {device}")
        if hf_token:
            print("🔑 HuggingFace Access Token provided.")
        print(f"🌐 Server running at http://{host}:{port}")
        
        vram = get_vram_usage(device)
        if vram["total_mb"] > 0:
            print(f"💾 VRAM Allocated: {vram['allocated_mb']}MB / {vram['total_mb']}MB")
            
        import threading
        from .server.app import init_model

        # Start model loading in background thread so web server opens instantly
        loader_thread = threading.Thread(
            target=init_model,
            args=(model_name, device, hf_token),
            daemon=True
        )
        loader_thread.start()

        import uvicorn
        uvicorn.run("interplens.server.app:app", host=host, port=port, reload=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
