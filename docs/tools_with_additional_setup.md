# Tools with additional setup

Some tools require additional setup

## Tools with additional large data files

Some tools require large files that are currently kept in the `thinkingbox-data` repository, under `support/`.

Tools:
- airline_tau_bench
- contoso_tools
- dataverse_v2 (dvsimple)
- fno_v1
- universal_search_tool

For these tool to be able to find those files, export the `THINKINGBOX_DATA`
environment variable pointing to the root of `thinkingbox-data`

```bash
export THINKINGBOX_DATA="/path/to/thinkingbox-data"
cd /path/to/thinkingbox
./scripts/background_tasks.sh
# or tb mcp-start
```

## TypeSense

**Needed by**

- search_sources

**Setup**

```bash
# Install into <virtualenv>/.thinkingbox and link to the virtualenv's bin directory

source ~/venv/bin/activate  # replace with your virtualenv
./scripts/install_typesense.sh
```

**Run**

```bash
# This script will start the MCP Session Proxy (tb mcp-start) and TypeSense in the background.
./scripts/background_tasks.sh

# Ctrl+C to stop
```

**Pre-index Knowledge Base**

You can pre-index a Knowledge Base directory into a stable Typesense collection and reuse it across runs.

The indexing script stores full documents (not pre-split snippets) and the search tool generates contextual snippets dynamically based on search queries.

```bash
# Set environment variables
cd /path/to/thinkingbox-data

# Index the documents
python scripts/create_kb_snapshot.py --collection "search_tool" \
    --docs-dir support/knowledge_base/kb_docs \
    --snapshot-dir support/knowledge_base/kb_snapshot
```

The script will:
- Walk through all files in the support/knowledge_base/kb_docs directory recursively
- Index full document content (not pre-split snippets)
- Store documents with deterministic IDs for efficient updates
- Skip binary files and hidden files (starting with '.')

It will produce a json snapshot for each source in the output directory


**Load pre-indexed documents**

Use `scripts/import_typesense_snapshots.py` to load snapshots recursively from a directory into a running typesense instance.

```bash
python scripts/import_typesense_snapshots.py --port 8108 --collection search_tool --inputs /path/to/snapshots
```

For a test search query

```bash
python -m thinkingbox.tools.toolslib.search_sources --port 8108 --collection search_tool -n 3 -q "deep neural networks" -k "neural network, deep learning"
```

## Embeddings Server

**Needed by**

- contoso_tools

**Setup**

Note: this requires the thinkingbox-data repository

```bash
cd /path/to/thinkingbox-data

# download required HF models to ./support/models
./scripts/download_hf_models.sh
```

**Run**

```bash
cd /path/to/thinkingbox-data

python -m thinkingbox.services.embeddings_hf_simple --model ./support/models/intfloat/e5-base-v2

# quick check that it works
curl http://127.0.0.1:7112/v1/embeddings -d '{"model": "default", "input": "A string"}'
curl http://127.0.0.1:7112/openai/deployments/default/embeddings -d '{"input": "A string"}'
```

Also make sure to export `THINKINGBOX_DATA`.
