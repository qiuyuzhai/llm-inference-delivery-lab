import argparse

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
        "stream": True,
        "max_tokens": 512,
    }
    if payload["model"] is None:
        payload.pop("model")

    with httpx.stream(
        "POST",
        f"{args.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        json=payload,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()
