.PHONY: setup up down ingest query eval

setup:
	python -m venv venv
	./venv/Scripts/activate && pip install -r requirements.txt

up:
	docker-compose up -d

down:
	docker-compose down

ingest:
	python -m ingestion.pipeline

query:
	python -m api.main

eval:
	python -m evaluation.benchmark
