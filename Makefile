.PHONY: test lint gateway benchmark compose-up compose-down

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check gateway benchmark tests

gateway:
	.venv/bin/python -m uvicorn gateway.app.main:app --host 0.0.0.0 --port 8080

benchmark:
	.venv/bin/python -m benchmark.harness.cli --base-url http://localhost:8080 --workload benchmark/workloads/knowledge_qa.jsonl --concurrency 1 --requests 3 --output benchmark/results/smoke.json

compose-up:
	docker compose -f deploy/docker-compose/docker-compose.yml up --build

compose-down:
	docker compose -f deploy/docker-compose/docker-compose.yml down
