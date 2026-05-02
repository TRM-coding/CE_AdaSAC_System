#!/usr/bin/env python3
import argparse
import shlex
import socket
import subprocess
import sys
import threading


def pipe(src, dst, label, stop):
    try:
        while not stop.is_set():
            data = src.recv(1 << 20) if hasattr(src, "recv") else src.read(1 << 20)
            if not data:
                break
            if hasattr(dst, "sendall"):
                dst.sendall(data)
            else:
                dst.write(data)
                dst.flush()
    except BrokenPipeError:
        pass
    finally:
        stop.set()
        if not hasattr(dst, "sendall"):
            try:
                dst.close()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Bridge local TCP layer-coop client to Android layer_mobile_server stdio over adb shell.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18100)
    parser.add_argument("--device-dir", default="/data/local/tmp/CE_Ada")
    parser.add_argument("--device-model", default="/data/local/tmp/CE_Ada/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf")
    parser.add_argument("--split", type=int, default=14)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", default="")
    args = parser.parse_args()

    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((args.listen_host, args.listen_port))
    listen.listen(1)
    print(f"adb-stdio-bridge listening on {args.listen_host}:{args.listen_port}", flush=True)

    client, addr = listen.accept()
    print(f"client connected {addr}", flush=True)

    remote_log = f"{args.device_dir}/layer_mobile_server_stdio.log"
    remote_cmd = (
        f"cd {shlex.quote(args.device_dir)} && "
        f"LD_LIBRARY_PATH=. taskset -a ff ./layer_mobile_server {shlex.quote(args.device_model)} 0 {args.split} {args.threads} stdio "
        f"2>{shlex.quote(remote_log)}"
    )
    adb_cmd = [args.adb]
    if args.serial:
        adb_cmd += ["-s", args.serial]
    adb_cmd += ["shell", remote_cmd]
    print("starting:", " ".join(shlex.quote(x) for x in adb_cmd), flush=True)

    proc = subprocess.Popen(adb_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    stop = threading.Event()
    threads = [
        threading.Thread(target=pipe, args=(client, proc.stdin, "client-to-adb", stop), daemon=True),
        threading.Thread(target=pipe, args=(proc.stdout, client, "adb-to-client", stop), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        client.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    client.close()
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    err = proc.stderr.read()
    if err:
        sys.stderr.buffer.write(err)


if __name__ == "__main__":
    main()
