# -*- coding: utf-8 -*-
"""
verify_speaking.py — 验证「不同输入 → 不同输出」

用 subprocess 驱动运行时，起一个全新实例（临时记忆文件，不碰真实存档），
依次喂入 6 个话题差异大的句子，每句读到下一个「师父> 」提示符为止，
从每轮输出里抽出回复正文（去掉 〔…〕 元信息行和提示符），
打印「输入 → 回复正文」对照表，最后检查是否有两句正文完全一样。

用法（在项目根目录运行）：
    python verify_speaking.py

退出码：0 = 6 句回复各不相同；1 = 有重复或运行中断。
"""

import os
import queue
import subprocess
import sys
import threading
import time

# ── 配置 ─────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TMP_FILE = "tmp_verify.luoxiaohei"   # 临时记忆文件，跑完即删
PROMPT = "师父> "                     # 主循环的输入提示符
META_MARK = "〔"                      # 元信息行标记（不是正文）

TEST_INPUTS = [
    "你好小黑",
    "小鱼干好吃吗",
    "你认识手机吗",
    "师父给你买了什么",
    "你喜欢吃什么",
    "讲一个笑话",
]

STARTUP_TIMEOUT = 600    # 首次启动要投喂种子文件，给足时间
TURN_TIMEOUT = 180       # 单轮对话最长等待秒数
SHUTDOWN_TIMEOUT = 60    # 结束时等子进程自己保存退出


# ── 子进程输出读取 ───────────────────────────────────────

class StreamReader:
    """后台线程把子进程输出逐字符搬进队列，主线程带超时取用。

    不用 select，Windows 的管道上也能跑。
    """

    def __init__(self, stream):
        self._chars = queue.Queue()
        self._thread = threading.Thread(
            target=self._pump, args=(stream,), daemon=True,
        )
        self._thread.start()

    def _pump(self, stream):
        while True:
            ch = stream.read(1)
            if ch == "":                 # 子进程关闭了输出
                self._chars.put(None)
                return
            self._chars.put(ch)


def read_until_prompt(reader, timeout):
    """逐字符读，直到末尾出现「师父> 」；返回提示符之前的文本。"""
    deadline = time.time() + timeout
    buf = []
    tail = ""
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("等待提示符超时，已收到：\n" + "".join(buf))
        try:
            ch = reader._chars.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError("等待提示符超时，已收到：\n" + "".join(buf))
        if ch is None:
            raise EOFError("子进程提前退出，已收到：\n" + "".join(buf))
        buf.append(ch)
        tail = (tail + ch)[-len(PROMPT):]
        if tail == PROMPT:
            return "".join(buf)[:-len(PROMPT)]


# ── 正文抽取 ─────────────────────────────────────────────

def extract_body(turn_text):
    """从一轮输出中抽出回复正文。

    正文 = 紧跟在输入之后、第一段不带〔标记的非空文本；
    〔…〕 元信息行（疲劳、自动保存、思考耗时等）一律跳过。
    """
    body_lines = []
    started = False
    for raw in turn_text.splitlines():
        line = raw.strip()
        if not started:
            if line and META_MARK not in line:
                started = True
                body_lines.append(line)
        else:
            if not line or META_MARK in line:
                break
            body_lines.append(line)
    return "\n".join(body_lines)


# ── 临时文件清理 ─────────────────────────────────────────

def clean_tmp_files():
    """删掉临时记忆文件及其备份，保证每次都是全新实例。"""
    for name in os.listdir(PROJECT_ROOT):
        if name.startswith(TMP_FILE):
            try:
                os.remove(os.path.join(PROJECT_ROOT, name))
            except OSError:
                pass


# ── 主流程 ───────────────────────────────────────────────

def main():
    os.chdir(PROJECT_ROOT)
    clean_tmp_files()

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"    # 子进程统一用 UTF-8 输出

    cmd = [
        sys.executable, "-u", "-m", "src.runtime.main",
        "--file", TMP_FILE, "--new",
    ]
    print("启动子进程：" + " ".join(cmd))
    print("（-u 让子进程输出不缓存，这样才能逐字符读到提示符）")

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,        # 报错也并进输出，方便定位
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    results = []        # [(输入, 正文, 本轮耗时)]
    error = None
    try:
        reader = StreamReader(proc.stdout)
        read_until_prompt(reader, STARTUP_TIMEOUT)   # 启动问候 + 首个提示符
        print("实例已启动（全新记忆），开始逐句喂入……\n")

        for sentence in TEST_INPUTS:
            t0 = time.time()
            proc.stdin.write(sentence + "\n")
            proc.stdin.flush()
            chunk = read_until_prompt(reader, TURN_TIMEOUT)
            elapsed = time.time() - t0
            results.append((sentence, extract_body(chunk), elapsed))
            print(f"  已喂 [{len(results)}/{len(TEST_INPUTS)}] {sentence}"
                  f"（{elapsed:.1f} 秒）")

    except (TimeoutError, EOFError) as e:
        error = str(e)
    except BrokenPipeError:
        error = "子进程管道断裂（可能已提前退出）"
    finally:
        # 关掉 stdin，让它自己走保存退出流程；退不掉就强杀
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=SHUTDOWN_TIMEOUT)
        except Exception:
            proc.kill()
        clean_tmp_files()

    # ── 对照表 ──
    print()
    print("=" * 56)
    print("输入 → 回复正文 对照表")
    print("=" * 56)
    for i, (sentence, body, elapsed) in enumerate(results, 1):
        print(f"[{i}] 输入：{sentence}（{elapsed:.1f} 秒）")
        lines = body.splitlines() if body else ["（没抽到正文）"]
        for line in lines:
            print(f"    回复：{line}")
        print()

    # ── 判定：有没有两句正文完全一样 ──
    print("=" * 56)
    if error is not None:
        print("运行中断：" + error)
        print("=" * 56)
        return 1
    if len(results) < len(TEST_INPUTS):
        print(f"只拿到 {len(results)}/{len(TEST_INPUTS)} 句回复，无法判定。")
        print("=" * 56)
        return 1

    groups = {}
    for i, (sentence, body, _) in enumerate(results, 1):
        groups.setdefault(body, []).append((i, sentence))
    duplicates = {b: items for b, items in groups.items() if len(items) > 1}

    if not duplicates:
        print(f"结论：{len(results)} 句输入的回复正文各不相同。")
        print("「不同输入 → 不同输出」验证通过，没有卡死在同一条回复上。")
        print("=" * 56)
        return 0

    print(f"结论：发现 {len(duplicates)} 组正文完全一样的回复：")
    for body, items in duplicates.items():
        tags = "、".join(f"[{i}]{s}" for i, s in items)
        preview = body[:60] + ("……" if len(body) > 60 else "")
        print(f"  相同输入：{tags}")
        print(f"  相同正文：{preview if body else '（空）'}")
    print("=" * 56)
    return 1


if __name__ == "__main__":
    sys.exit(main())
