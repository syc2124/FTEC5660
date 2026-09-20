#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    prompt_text = ('列出这张小票上的最终支付额、小计、所有折扣额、四舍五入额和商品明细。以json格式输出，不要遗漏任何一行，不要有任何前后解释文字，所有折扣额取正数部分、四舍五入额保留原符号。（小票上有若干促销折扣行，折扣行可能以多种形式出现，凡是该行带负号金额、且不是商品单价的，都要计入折扣额，例如：'
                    'Buy N Save $X（促销）、MB PRICE（会员价）、x% OFF（百分比折扣）、'
                    'MB APP UPGRADE / App Upgrad$（App 折扣）、COUPON（优惠券）、'
                    '包装變形 / 包裝破損 / packaging damage（包装损坏补偿）。'
                    '特别注意"包装變形""包裝破損"这类行，它们常以中文出现在商品名称下方，金额同样带负号，'
                    '也属于折扣，不要遗漏。折扣金额一律取该行最右侧带负号的数字（例如 -$6.00）。行内左侧的说明文字（如 Buy 2 Save $6）可能包含数字，那些不是金额，请忽略。'
                    '如果同一行上的两个数字看起来不一致，以最右侧带负号的数字为准。）'
                    '输出样例：{ \"final_amount\": \"123.40\", \"subtotal\": \"150.00\", \"discounts\": [\"26.60\"], \"rounding\": \"0.01\", \"items\": [{\"code\": \"001\", \"name\": \"商品1\", \"quantity\": 2, \"amount\": \"10.00\"}] }')
    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        timeout=120,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("human", [
            {"type": "text", "text": "{instructions}"},
            {"type": "image_url", "image_url": {"url": "{receipt}"}},
        ]),
    ]).partial(instructions=prompt_text)

    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    def money(value):
        return Decimal(str(value).replace("$", "").replace(",", "").strip())
        # return Decimal(str(value))

    def is_consistent(data) -> bool:
        """这张小票的字段是否自洽"""
        try:
            subtotal = money(data["subtotal"])
            final_amount = money(data["final_amount"])
            discounts = [money(d) for d in data.get("discounts") or []]
            rounding = money(data.get("rounding", 0))
            items = [money(i["amount"]) for i in data.get("items") or []]
        except Exception:
            return False
        return (
                bool(items)
                and sum(items) == subtotal + sum(discounts)  # 校验 A
                and subtotal + rounding == final_amount  # 校验 B
        )

    # ① 每张图变成一次调用的输入
    inputs = [{"receipt": image_data_url(path)} for path in images]

    # ② 并行跑（7 张图 → 7 次请求，最多同时 4 个）
    # results = chain.batch(inputs, config={"max_concurrency": 4})
    results = []
    for attempt in range(3):
        try:
            results = chain.batch(inputs, config={"max_concurrency": 4})
            break
        except Exception as exc:
            print(f"[warn] batch attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")
            continue

    # 找出不自洽的，重读一次
    bad = []
    for i, raw in enumerate(results):
        try:
            data = json.loads(response_text(raw))
        except Exception:
            bad.append(i)  # 解析失败也重试
            continue
        if not is_consistent(data):
            bad.append(i)

    if bad:
        try:
            repair = chain.batch([inputs[i] for i in bad], config={"max_concurrency": 4})
            for i, raw in zip(bad, repair):
                results[i] = raw
        except Exception:
            pass

    # ③ 逐张解析并累加
    total_paid = Decimal("0")  # 回答 QUERY_1 用
    total_without_discount = Decimal("0")  # 回答 QUERY_2 用

    for image, result in zip(images, results):
        try:
            data = json.loads(response_text(result))
            subtotal = money(data["subtotal"])
            final_amount = money(data["final_amount"])
            discounts = [money(d) for d in data.get("discounts") or []]
        except Exception:
            print(f"[detail] {image.name} PARSE-FAIL")
            continue

        print(
            f"[detail] {image.name} final={final_amount} subtotal={subtotal} "
            f"sum_disc={sum(discounts)} q2={subtotal + sum(discounts)} "
            f"discs={' '.join(str(d) for d in discounts)}"
        )

        # 计算账单花销、不打折花销
        total_paid += final_amount
        total_without_discount += subtotal + sum(discounts)

    # ④ 返回
    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
