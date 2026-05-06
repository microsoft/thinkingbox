# How to run ThinkingBox tl;dr

## Prerequisites

- Access to our GitHub repo (if you're reading this...)
- WSL, install `azure-cli`, `uv`
- An AOAI (Azure OpenAI) deployment


## Authenticate with GitHub

SSH is the most reliable.

**Get SSH key**

Check if you have an SSH key

```bash
ls ~/.ssh
# id_rsa and id_rsa.pub, or ed25519 equivalent should be there
```

If you don't, generate one. SET A PASSWORD when prompted.

```bash
ssh-keygen
```

**Add SSH key to GitHub**

- Go to [https://github.com/settings/keys](https://github.com/settings/keys)
- Add new SSH key, paste the content of the `.pub` file in `~/.ssh`. NOT THE PRIVATE KEY.
- Next to the entry for the new key, select "Configure SSO" -> Authorize the microsoft organization

**Use SSH to clone**

Always clone with the SSH address, e.g.

```bash
git clone git@github.com:microsoft/thinkingbox.git
```


## Install and run TB

```bash
# login
az login

# Clone TB and tools+data
git clone git@github.com:microsoft/thinkingbox.git
git clone git@github.com:microsoft/thinkingbox-data.git

# Install TB and tools
uv venv --python 3.12 thinkingbox/.venv
source thinkingbox/.venv/bin/activate
uv pip install --config-settings editable-mode=compat -e 'thinkingbox[dev]'
uv pip install --config-settings editable-mode=compat -e thinkingbox-data/servers/thinkingbox_tools

# start MCP session proxy, leave it running
THINKINGBOX_DATA="thinkingbox-data" tb mcp-start --servers thinkingbox-data/servers/servers.yaml

# NEW TAB. activate venv
source thinkingbox/.venv/bin/activate

# Run a test
tb infer -c thinkingbox/config/config_o4mini.yaml --dataset thinkingbox-data/dataset --agent think --inputs thinkingbox-data/dataset/test_case/banking.py -r 1 -b 10 -o output.jsonl

# show results table
tb agg --concise output.jsonl

# show first conversation
tb pp --less output.jsonl --line 1
```


## Typesense

The example above does not need typesense, but a few other tools need it.

```bash
# Install in the virtualenv
source thinkingbox/.venv/bin/activate
thinkingbox/scripts/install_typesense.sh

# Start
mkdir -p /tmp/typesense/data && typesense-server --data-dir="/tmp/typesense/data" --api-key="Fake" --enable-cors
```


## AOAI

Default config LLM does not work?

**Option 1** you have access to B360-AI-LAB, set the default subscription with `az account set --subscription 199b3c58-a154-4c89-80c8-6bbc39e6bee0`

**Option 2** you don't have access, go get another LLM endpoint.

Go to [https://ai.azure.com](https://ai.azure.com) and find a LLM endpoint. Or create one.

Edit the endpoints in the config file (`tb infer -c <this-is-the-config-file>`)

```
# ...

orchestrator:
  type: thinkingbox
  agent_model:
    account_name: "your-aoai-account-name"
    deployment: "your-aoai-reasoning-deployment-name"
    # ...

judge_model:
  account_name: "your-aoai-account-name"
  deployment: "your-aoai-deployment-name"
  # ...

user_model:
  account_name: "your-aoai-account-name"
  deployment: "your-aoai-deployment-name"
  # ...
```

# What's next?

- [onboarding.md](onboarding.md): a more comprehensive version of this guide
- [tutorial.md](tutorial.md): write your first tool, scenario and test case
