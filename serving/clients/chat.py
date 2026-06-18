import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="change-me")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
        "max_tokens": 512,
    }
    if payload["model"] is None:
        payload.pop("model")

    response = httpx.post(
        f"{args.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
