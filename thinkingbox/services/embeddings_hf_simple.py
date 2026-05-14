# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
AOAI-compatible embeddings server for testing. Uses e5-base-v2. It can run on CPU

Limitations:
- does not support base64 mode
- requests are processed sequentially in a queue
- won't use GPU
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from typing import Literal

import torch
import uvicorn
from pydantic import BaseModel
from pydantic_core import to_json
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from transformers import AutoModel, AutoTokenizer


class AppConfig(BaseModel):
    model_path: str = ""
    model_name: str = "default"


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str = "default"
    dimensions: int = 0
    encoding_format: Literal["float"] = "float"  # TODO "base64" ?
    user: str = ""  # ignored


class Embedding(BaseModel):
    embedding: list[float]
    index: int
    object: Literal["embedding"] = "embedding"


class EmbeddingsResponse(BaseModel):
    data: list[Embedding]


class EmbeddingsE5:
    def __init__(self, model_path, cache_size=200):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.cache_size = cache_size
        self.cache = {}

    def _preprocess(self, text: str):
        return f"query: {text}"

    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def compute(self, text_inputs):
        outputs = [None] * len(text_inputs)
        inputs = []
        inputs_map = []
        for i, t in enumerate(text_inputs):
            t = self._preprocess(t)
            t_emb = self.cache.get(t)
            if t_emb is not None:
                outputs[i] = t_emb
            else:
                inputs.append(t)
                inputs_map.append(i)
        del text_inputs
        if inputs:
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            with torch.no_grad():
                batch_dict = self.tokenizer(
                    inputs,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                model_out = self.model(**batch_dict)
                embeddings = EmbeddingsE5.average_pool(
                    model_out.last_hidden_state, batch_dict["attention_mask"]
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings = list(embeddings.numpy())
            for k, i in enumerate(inputs_map):
                emb = embeddings[k]
                self.cache[inputs[k]] = emb
                outputs[i] = emb
        return outputs


def json_to_obj(Class, text: str):
    try:
        obj = json.loads(text.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("Not a JSON object")
        return Class(**obj)
    except (ValueError, TypeError, UnicodeDecodeError) as e:
        raise ValueError(str(e))


def error_response(msg):
    return JSONResponse({"error_message": msg}, status_code=400)


async def embeddings(request):
    payload = await request.body()
    try:
        obj: EmbeddingsRequest = json_to_obj(EmbeddingsRequest, payload)
        if obj.model not in (app_config.model_name, ""):
            raise ValueError("Invalid model")
    except ValueError as e:
        return error_response(str(e))

    if isinstance(obj.input, str):
        inputs = [obj.input]
    else:
        inputs = obj.input
    response_q = asyncio.Queue()
    await request.app.model_queue.put((inputs, response_q))
    embeddings = await response_q.get()
    if not embeddings:
        return error_response("Internal error")
    out = EmbeddingsResponse(data=[])
    for i, emb in enumerate(embeddings):
        out.data.append(Embedding(embedding=emb, index=i))
    return Response(to_json(out), status_code=200, media_type="application/json")


async def server_loop(q):
    try:
        model_path = app_config.model_path
        print(f"Loading model: {model_path}")
        model = EmbeddingsE5(model_path)
        print("Loaded")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    while True:
        out = None
        try:
            (text_inputs, response_q) = await q.get()
            out = model.compute(text_inputs)
        except Exception:
            traceback.print_exc()
        await response_q.put(out)


app = Starlette()
app_config = AppConfig()


@app.on_event("startup")
async def startup_event():
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("uvicorn.access").handlers[0].setFormatter(formatter)

    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7112)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--served_model_name", type=str, default="default")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise ValueError(f"Model directory not found: {args.model}")

    app_config.model_path = args.model
    app_config.model_name = args.served_model_name

    app.routes.extend(
        [
            Route("/v1/embeddings", embeddings, methods=["POST"]),
            Route(
                f"/openai/deployments/{app_config.model_name}/embeddings",
                embeddings,
                methods=["POST"],
            ),
        ],
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
    )


# Run the server using uvicorn
if __name__ == "__main__":
    main()
