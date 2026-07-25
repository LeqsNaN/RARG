"""
Model server client — connects to the persistent model_server.py.

Two modes:
  1. TCP mode (MODEL_SERVER_PORT set): connects to an externally-managed persistent server.
     Multiple agent processes share one server. Server stays alive between queries.
  2. Subprocess mode (default): spawns model_server.py as a child process with stdin/stdout
     JSON-line protocol. Server dies when the agent process exits.

Communicates via JSON-line protocol.
Provides high-level interfaces for embed_recall and rerank_matches.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import AgentConfig

logger = logging.getLogger(__name__)


class ModelServerClient:
    """
    Client for the persistent model_server.py process.

    If MODEL_SERVER_PORT is set, connects via TCP to an existing server.
    Otherwise, starts model_server.py as a subprocess (stdin/stdout mode).
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._socket: Optional[socket.socket] = None
        self._socket_rfile = None
        self._socket_wfile = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._started = False
        self._tcp_mode = False

    def ensure_started(self) -> None:
        """Start or connect to model_server if not already connected."""
        if self._started:
            if self._tcp_mode:
                # Check socket is still alive
                if self._socket is not None:
                    return
            else:
                # Check process is still alive
                if self.process and self.process.poll() is None:
                    return
        self._start_server()

    def _start_server(self) -> None:
        """Connect via TCP or launch model_server.py subprocess."""
        tcp_port = int(os.environ.get("MODEL_SERVER_PORT", "0"))
        if tcp_port > 0:
            self._connect_tcp(tcp_port)
        else:
            self._spawn_subprocess()

    def _connect_tcp(self, port: int) -> None:
        """Connect to an existing model_server via TCP."""
        self._tcp_mode = True
        logger.info(f"Connecting to model server at 127.0.0.1:{port}...")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)  # 30s connect timeout
        try:
            sock.connect(("127.0.0.1", port))
        except (ConnectionRefusedError, socket.timeout) as e:
            raise RuntimeError(
                f"Cannot connect to model server at 127.0.0.1:{port}: {e}\n"
                "Start it with: bash scripts/start_model_server.sh"
            )

        sock.settimeout(None)  # Back to blocking mode for reads
        self._socket = sock
        self._socket_rfile = sock.makefile('r', encoding='utf-8')
        self._socket_wfile = sock.makefile('w', encoding='utf-8')

        # Wait for ready signal (id=0)
        ready = self._read_response_tcp(timeout=30)
        if ready is None:
            raise RuntimeError("Model server TCP: no ready signal received")
        if "error" in ready:
            raise RuntimeError(f"Model server TCP startup error: {ready['error']}")

        logger.info("Model server TCP connection ready")
        self._started = True

    def _spawn_subprocess(self) -> None:
        """Launch model_server.py subprocess and wait for ready signal."""
        self._tcp_mode = False
        config = self.config

        # Determine python executable
        embed_python = os.environ.get("EMBED_PYTHON", sys.executable)

        cmd = [
            embed_python,
            config.model_server_script,
            "--index-dir", config.index_dir,
            "--embed-model-path", config.embed_model_path,
            "--qr-model-path", config.qr_model_path,
            "--qr-heads", config.qr_heads,
            "--device", config.device,
            "--corpus-dir", config.corpus_dir,
        ]
        if config.no_reranker:
            cmd.append("--no-reranker")

        logger.info(f"Starting model server: {' '.join(cmd[:4])}...")

        # Build env for subprocess
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )

        # Start stderr reader thread (logs to our logger)
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_thread.start()

        # Wait for ready signal
        ready = self._read_response_stdio(timeout=300)  # model loading can take a few minutes
        if ready is None:
            raise RuntimeError("Model server failed to send ready signal (timeout)")

        if "error" in ready:
            raise RuntimeError(f"Model server startup error: {ready['error']}")

        result = ready.get("result", {})
        load_time = result.get("load_time_s", "?")
        logger.info(f"Model server ready (load time: {load_time}s)")
        self._started = True

    def _read_stderr(self) -> None:
        """Read stderr from model_server and log it."""
        assert self.process and self.process.stderr
        for line in iter(self.process.stderr.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                logger.debug(f"[model_server] {text}")

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_request_tcp(self, method: str, params: Dict[str, Any]) -> int:
        """Send a JSON-line request via TCP socket."""
        assert self._socket_wfile is not None
        req_id = self._next_id()
        request = {"id": req_id, "method": method, "params": params}
        line = json.dumps(request, ensure_ascii=False) + "\n"
        self._socket_wfile.write(line)
        self._socket_wfile.flush()
        return req_id

    def _read_response_tcp(self, timeout: float = 120) -> Optional[Dict[str, Any]]:
        """Read a single JSON-line response from TCP socket."""
        assert self._socket_rfile is not None
        assert self._socket is not None
        self._socket.settimeout(timeout)
        try:
            line = self._socket_rfile.readline()
            if not line:
                return None
            return json.loads(line)
        except socket.timeout:
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from model_server TCP: {e}")
            return None
        finally:
            self._socket.settimeout(None)

    def _send_request_stdio(self, method: str, params: Dict[str, Any]) -> int:
        """Send a JSON-line request to model_server stdin."""
        assert self.process and self.process.stdin
        req_id = self._next_id()
        request = {"id": req_id, "method": method, "params": params}
        line = json.dumps(request, ensure_ascii=False) + "\n"
        self.process.stdin.write(line.encode("utf-8"))
        self.process.stdin.flush()
        return req_id

    def _read_response_stdio(self, timeout: float = 120) -> Optional[Dict[str, Any]]:
        """Read a single JSON-line response from model_server stdout."""
        assert self.process and self.process.stdout
        start = time.perf_counter()
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"Model server process died (exit code {self.process.returncode})")
            line = self.process.stdout.readline()
            if line:
                try:
                    return json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from model_server: {line[:200]}")
                    continue
            if time.perf_counter() - start > timeout:
                return None
            time.sleep(0.01)

    def request(self, method: str, params: Dict[str, Any], timeout: float = 120) -> Dict[str, Any]:
        """Send request and wait for response (thread-safe)."""
        with self._lock:
            self.ensure_started()
            if self._tcp_mode:
                self._send_request_tcp(method, params)
                response = self._read_response_tcp(timeout=timeout)
            else:
                self._send_request_stdio(method, params)
                response = self._read_response_stdio(timeout=timeout)

        if response is None:
            raise RuntimeError(f"Model server request timed out: {method}")
        if "error" in response:
            raise RuntimeError(f"Model server error ({method}): {response['error']}")
        return response.get("result", {})

    # =========================================================================
    # High-level interfaces
    # =========================================================================

    def embed_recall(
        self,
        query: str,
        top_k: int = 5000,
        scope_dir: str = ".",
        scope_size_prefix: str = "",
        sample_id: str = "",
        paragraph_rerank_model: str = "qr",
        paragraph_rerank_doc_limit: int = 0,
    ) -> Dict[str, Any]:
        """
        Semantic recall: search FAISS index and return scope file + paragraph-ranked snippets.

        Returns dict with keys: output, scope_file, count, elapsed_ms
        """
        return self.request("embed_recall", {
            "query": query,
            "top_k": top_k,
            "scope_dir": scope_dir,
            "scope_size_prefix": scope_size_prefix,
            "sample_id": sample_id,
            "paragraph_rerank_model": paragraph_rerank_model,
            "paragraph_rerank_doc_limit": paragraph_rerank_doc_limit,
        }, timeout=120)

    def rerank_matches(
        self,
        query: str,
        matches: List[Dict[str, Any]],
        timeout: float = 60,
        rerank_model: str = "qr",
    ) -> Dict[str, Any]:
        """
        Rerank grep matches using the configured reranker backend.

        Args:
            query: Natural language query for relevance scoring
            matches: List of {"text": str, "file": str, "line": int}
            timeout: Request timeout in seconds
            rerank_model: "qr" for QRReranker, "emb" for primary embedding model,
                "qwen3emb" for a separate Qwen3 embedding reranker

        Returns dict with keys: ranked_indices, elapsed_ms
        """
        return self.request("rerank_matches", {
            "query": query,
            "matches": matches,
            "rerank_model": rerank_model,
        }, timeout=timeout)

    def shutdown(self) -> None:
        """Gracefully shut down / disconnect."""
        if not self._started:
            return

        if self._tcp_mode:
            # TCP mode: just close socket, don't shut down remote server
            self._close_tcp()
        else:
            # Subprocess mode: send shutdown and kill
            if self.process and self.process.poll() is None:
                try:
                    self.request("shutdown", {}, timeout=10)
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            logger.info("Model server shut down")

        self._started = False

    def _close_tcp(self) -> None:
        """Close TCP connection."""
        if self._socket_rfile:
            try:
                self._socket_rfile.close()
            except Exception:
                pass
        if self._socket_wfile:
            try:
                self._socket_wfile.close()
            except Exception:
                pass
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None
        self._socket_rfile = None
        self._socket_wfile = None
        logger.info("Model server TCP connection closed")

    def __del__(self):
        self.shutdown()
