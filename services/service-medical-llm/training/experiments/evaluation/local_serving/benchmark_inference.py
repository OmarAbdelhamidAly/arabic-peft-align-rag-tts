import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple local inference benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    endpoint = f"{args.base_url}/v1/chat/completions"
    durations = []
    failures = 0

    for i in range(args.requests):
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Explain diabetes management in simple Arabic. Request #{i + 1}",
                }
            ],
            "temperature": 0.2,
            "max_tokens": args.max_tokens,
        }

        start = time.perf_counter()
        try:
            result = post_json(endpoint, payload, args.timeout)
            elapsed = (time.perf_counter() - start) * 1000.0
            durations.append(elapsed)

            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            print(
                f"[{i + 1}/{args.requests}] OK {elapsed:.1f} ms "
                f"(prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens})"
            )
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failures += 1
            print(f"[{i + 1}/{args.requests}] FAIL {exc}")

    success_count = len(durations)
    print("\n=== Benchmark Summary ===")
    print(f"Total requests: {args.requests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failures}")

    if durations:
        print(f"Avg latency: {statistics.mean(durations):.1f} ms")
        print(f"Median latency: {statistics.median(durations):.1f} ms")
        print(f"P95 latency: {percentile(durations, 95):.1f} ms")
        print(f"Min latency: {min(durations):.1f} ms")
        print(f"Max latency: {max(durations):.1f} ms")


if __name__ == "__main__":
    main()
