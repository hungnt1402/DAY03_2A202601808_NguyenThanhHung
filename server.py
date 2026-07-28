"""
🌐 WEB APP SERVER & API FOR SMART RETURN ASSISTANT
Chạy file này để khởi chạy giao diện UI & Live Backend API: python server.py

Server phục vụ index.html và mở API để giao diện gọi thẳng vào Agent thật
trong src/ (tool thật + LLM thật), thay vì mô phỏng bằng JavaScript.

Endpoint:
  GET  /api/info  -> thông tin Provider/Model đang dùng
  POST /api/chat  -> hỏi Agent (mode: "react" mặc định, hoặc "baseline")
"""

import http.server
import json
import os
import socketserver
import sys
import time
import webbrowser

from dotenv import load_dotenv

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(DIRECTORY, "src"))

# Đảm bảo in Tiếng Việt / Emoji không lỗi trên Windows Console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv(override=True)

# Tái sử dụng đúng các thành phần Role 2/3/4 đã xây, không viết lại logic
from providers import get_llm_provider, MockProvider
from prompts import CHATBOT_BASELINE_PROMPT, REACT_SYSTEM_PROMPT, MAX_ITERATIONS
from app import (
    build_final_answer,
    execute_tool,
    extract_order_id,
    parse_action,
    plan_next_step,
    safe_json_loads,
)

PORT = 8000


def provider_label(provider) -> str:
    return f"{provider.__class__.__name__} ({getattr(provider, 'model_name', 'Offline Mock Mode')})"


def run_baseline_api(user_query: str) -> dict:
    """Chatbot Baseline: gọi LLM thật, không dùng bất kỳ tool nào."""
    started = time.time()
    provider = get_llm_provider()
    answer = provider.generate(user_query, system_prompt=CHATBOT_BASELINE_PROMPT)

    return {
        "provider": provider_label(provider),
        "is_mock": isinstance(provider, MockProvider),
        "mode": "baseline",
        "route": "chatbot-path",
        "query": user_query,
        "iterations_used": 1,
        "steps": [{"step": 1, "thought": "", "final": answer}],
        "final_answer": answer,
        "latency_ms": int((time.time() - started) * 1000),
    }


def run_live_agent_api(user_query: str) -> dict:
    """
    Thực thi ReAct Agent và trả về mảng các bước Thought, Action, Observation, Final Answer.

    Áp dụng Hybrid Decision Flow đã thống nhất ở Mốc 4 (docs/trace_eval.md §4):
    câu hỏi không kèm mã đơn là câu hỏi chính sách chung -> đi Chatbot path,
    vì không cần bằng chứng riêng của đơn hàng nào.
    """
    if not extract_order_id(user_query):
        result = run_baseline_api(user_query)
        result["route"] = "chatbot-path (hybrid: câu hỏi chung, không có mã đơn)"
        return result

    load_dotenv(override=True)
    started = time.time()
    provider = get_llm_provider()
    is_mock = isinstance(provider, MockProvider)

    observations = []
    action_history = set()
    steps = []
    finished = False

    for step in range(1, MAX_ITERATIONS + 1):
        if not is_mock:
            history_prompt = f"User Question: {user_query}\n"
            for obs in observations:
                history_prompt += f"\nAction: {obs['tool_name']}{obs['args']}\nObservation: {obs['raw']}\n"

            agent_output = provider.generate(history_prompt, system_prompt=REACT_SYSTEM_PROMPT) or ""
            # Nếu LLM báo lỗi API hoặc trả về rỗng -> fallback planner tất định
            if not agent_output or (agent_output.startswith("[") and ("Exception" in agent_output or "Error" in agent_output)):
                agent_output = plan_next_step(user_query, observations)
        else:
            agent_output = plan_next_step(user_query, observations)

        # Trích xuất Thought
        thought = ""
        if "Thought:" in agent_output:
            thought_part = agent_output.split("Thought:")[1]
            if "Action:" in thought_part:
                thought = thought_part.split("Action:")[0].strip()
            elif "Final Answer:" in thought_part:
                thought = thought_part.split("Final Answer:")[0].strip()
            else:
                thought = thought_part.strip()
        else:
            thought = agent_output

        # Trích xuất Final Answer
        if "Final Answer:" in agent_output:
            final = agent_output.split("Final Answer:")[1].strip()
            steps.append({"step": step, "thought": thought, "final": final})
            finished = True
            break

        # Parse Action
        parsed = parse_action(agent_output)
        if not parsed:
            obs_text = '{"status":"error","message":"Không parse được Action."}'
            observations.append({"tool_name": "parser", "args": [], "raw": obs_text})
            steps.append({
                "step": step,
                "thought": thought,
                "action": "unknown",
                "observation": obs_text,
                "toolName": "parser",
                "toolStatus": "error",
                "guardrail": "Parser Action",
            })
            continue

        tool_name, args = parsed
        action_key = (tool_name, tuple(args))

        if action_key in action_history:
            steps.append({
                "step": step,
                "thought": "Agent bị lặp lại cùng một Action với cùng tham số.",
                "final": "Mình đang bị kẹt ở cùng một bước xử lý, nên sẽ dừng an toàn.",
                "guardrail": "Repeated Action Detection",
            })
            finished = True
            break

        action_history.add(action_key)
        tool_started = time.time()
        obs_text = execute_tool(tool_name, args)
        tool_latency = int((time.time() - tool_started) * 1000)
        obs_data = safe_json_loads(obs_text)
        status = obs_data.get("status", "success") if isinstance(obs_data, dict) else "success"

        observations.append({
            "tool_name": tool_name,
            "args": args,
            "raw": obs_text,
            "data": obs_data,
        })

        steps.append({
            "step": step,
            "thought": thought,
            "action": f"{tool_name}{args}",
            "observation": obs_text,
            "toolName": tool_name,
            "toolStatus": status,
            "latency_ms": tool_latency,
        })

    if not finished:
        # Hết lượt lặp nhưng vẫn còn bằng chứng đã thu được: tổng hợp câu trả lời
        # từ Observation cuối thay vì bỏ đi và trả lời cụt cho người dùng.
        if observations:
            fallback = build_final_answer(user_query, observations).replace("Final Answer:", "").strip()
        else:
            fallback = "Mình chưa thể hoàn tất yêu cầu trong giới hạn xử lý an toàn."
        steps.append({
            "step": MAX_ITERATIONS,
            "thought": f"Đã đạt giới hạn tối đa {MAX_ITERATIONS} bước, chốt câu trả lời từ bằng chứng đã có.",
            "final": fallback,
            "guardrail": f"MAX_ITERATIONS = {MAX_ITERATIONS}",
        })

    final_answer = next((s["final"] for s in reversed(steps) if s.get("final")), None)

    return {
        "provider": provider_label(provider),
        "is_mock": is_mock,
        "mode": "react",
        "route": "react-agent-path",
        "query": user_query,
        # Đếm số vòng lặp Agent thật sự dùng, không tính bước tổng hợp cuối
        "iterations_used": min(len([s for s in steps if s.get("action")]) + 1, MAX_ITERATIONS),
        "steps": steps,
        "final_answer": final_answer,
        "latency_ms": int((time.time() - started) * 1000),
    }


class LiveHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/favicon.ico":
            # Trả rỗng thay vì 404 để console không báo lỗi rác khi demo
            self.send_response(204)
            self.end_headers()
            return

        if self.path == "/api/info":
            load_dotenv(override=True)
            p = get_llm_provider()
            self._send_json({
                "provider": provider_label(p),
                "is_mock": isinstance(p, MockProvider),
                "max_iterations": MAX_ITERATIONS,
            })
            return

        super().do_GET()

    def do_POST(self):
        # /api/ask là tên cũ, giữ lại để không vỡ giao diện đang gọi endpoint đó
        if self.path not in ("/api/chat", "/api/ask"):
            self._send_json({"error": "Endpoint không tồn tại."}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"Body không hợp lệ: {exc}"}, status=400)
            return

        query = (data.get("query") or "").strip()
        mode = data.get("mode") or "react"
        if not query:
            self._send_json({"error": "Thiếu nội dung câu hỏi."}, status=400)
            return

        try:
            result = run_baseline_api(query) if mode == "baseline" else run_live_agent_api(query)
        except Exception as exc:  # Không để server chết vì 1 câu hỏi lỗi
            self._send_json({"error": f"Lỗi xử lý phía server: {exc}"}, status=500)
            return

        self._send_json(result)


class ThreadedServer(socketserver.ThreadingTCPServer):
    """Cho phép xử lý nhiều request song song (LLM call có thể mất vài giây)."""

    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        """
        Người dùng đóng tab hoặc F5 giữa lúc chờ LLM là chuyện bình thường.
        Nuốt lỗi ngắt kết nối để console không bị ngập traceback đỏ khi demo.
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print("==================================================")
    print("🚀 ĐANG KHỞI CHẠY GIAO DIỆN SMART RETURN ASSISTANT UI")
    print("==================================================")
    print(f"🔌 LLM Provider: {provider_label(get_llm_provider())}")
    print(f"🔗 Mở trình duyệt tại: {url}")
    print("Press Ctrl+C to stop the web server.\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    with ThreadedServer(("", PORT), LiveHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Đã dừng Web Server.")
            sys.exit(0)
