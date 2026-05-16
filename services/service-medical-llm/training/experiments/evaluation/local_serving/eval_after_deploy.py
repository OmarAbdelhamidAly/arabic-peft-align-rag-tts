import argparse
import json
import urllib.error
import urllib.request


TEST_PROMPTS = [
    "اشرح ضغط الدم المرتفع للمريض بلغة بسيطة.",
    "ما النصائح الأساسية لمريض السكري من النوع الثاني؟",
    "متى يجب التوجه للطوارئ عند ألم الصدر؟",
    "اشرح أهمية الالتزام بجرعة الدواء اليومية.",
    "كيف نفرق بين أعراض الزكام والإنفلونزا بشكل عام؟",
]


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run smoke evaluation on deployed chat endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    endpoint = f"{args.base_url}/v1/chat/completions"

    passed = 0
    failed = 0

    print("=== Post-Deploy Smoke Evaluation ===")
    for idx, prompt in enumerate(TEST_PROMPTS, start=1):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 120,
        }
        try:
            result = post_json(endpoint, payload, args.timeout)
            answer = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if len(answer) >= 20:
                passed += 1
                print(f"[{idx}] PASS - response length={len(answer)}")
            else:
                failed += 1
                print(f"[{idx}] FAIL - response too short")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failed += 1
            print(f"[{idx}] FAIL - {exc}")

    print("\n=== Evaluation Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if failed == 0:
        print("Status: READY FOR DEMO")
    else:
        print("Status: NEEDS REVIEW")


if __name__ == "__main__":
    main()
