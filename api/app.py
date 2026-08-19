"""Flask REST API wiring the preprocessing, embedding, and search layers together."""

import uuid

import pandas as pd
from elasticsearch import Elasticsearch, NotFoundError
from flask import Flask, jsonify, request

from embed.model import get_embedder
from preprocess.clean import bucket_for_years
from search.es_client import INDEX_NAME, build_client
from search.indexing import ensure_index, index_advisors
from search.query import hybrid_search


def create_app(es_client: Elasticsearch | None = None, index_name: str = INDEX_NAME) -> Flask:
    app = Flask(__name__)
    client = es_client or build_client()
    ensure_index(client, index_name=index_name)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/advisors")
    def create_advisor():
        body = request.get_json(force=True)
        for field in ("name", "bio"):
            if not body.get(field):
                return jsonify({"error": f"'{field}' is required"}), 400

        advisor_id = str(uuid.uuid4())
        years = body.get("years_experience")
        row = pd.DataFrame(
            [
                {
                    "id": advisor_id,
                    "name": body["name"],
                    "bio": body["bio"],
                    "tags": body.get("tags", []),
                    "years_experience": years,
                    "experience_bucket": bucket_for_years(years),
                }
            ]
        )

        embedder = get_embedder()
        embeddings = embedder.embed_texts([body["bio"]])
        index_advisors(client, row, embeddings, index_name=index_name)

        return jsonify({"advisor_id": advisor_id}), 201

    @app.get("/advisors/<advisor_id>")
    def get_advisor(advisor_id: str):
        try:
            doc = client.get(index=index_name, id=advisor_id, source_excludes=["embedding"])
        except NotFoundError:
            return jsonify({"error": "advisor not found"}), 404
        return jsonify(doc["_source"])

    @app.post("/match")
    def match():
        body = request.get_json(force=True)
        query_text = body.get("query")
        if not query_text:
            return jsonify({"error": "'query' is required"}), 400
        k = int(body.get("k", 5))

        embedder = get_embedder()
        query_embedding = embedder.embed_text(query_text)
        results = hybrid_search(client, query_text, query_embedding, k=k, index_name=index_name)

        return jsonify({"results": results})

    return app
