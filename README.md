<div align="center">
  <a href="https://docs.langchain.com/oss/python/langchain/overview">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <img alt="LangChain Logo" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>The agent engineering platform.</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/langchain" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/langchain" target="_blank"><img src="https://img.shields.io/pepy/dt/langchain" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/langchain/#history" target="_blank"><img src="https://img.shields.io/pypi/v/langchain?label=%20" alt="Version"></a>
  <a href="https://x.com/langchain" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

<br>

LangChain is a framework for building agents and LLM-powered applications. It helps you chain together interoperable components and third-party integrations to simplify AI application development — all while future-proofing decisions as the underlying technology evolves.

> [!NOTE]
> Looking for the JS/TS library? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quickstart

```bash
pip install langchain
# or
uv add langchain
```

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-5.4")
result = model.invoke("Hello, world!")
```

If you're looking for more advanced customization or agent orchestration, check out [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), our framework for building controllable agent workflows.

> [!TIP]
> For developing, debugging, and deploying AI agents and LLM applications, see [LangSmith](https://docs.langchain.com/langsmith/home).

## Where to Contribute

If you are new to this monorepo, start with `AGENTS.md`, `libs/README.md`, and the `README.md` plus `pyproject.toml` for the package you expect to touch. In this checkout, the repo-local `docs/` directory is empty, so contributor guidance currently lives in these Markdown files and in package-local `Makefile` targets.

- `libs/core`: Base abstractions and shared runtime primitives. Edit this for changes to runnables, messages, prompts, tools, callbacks, language model interfaces, or serialization.
- `libs/langchain_v1`: The actively maintained `langchain` package. Edit this for new top-level agent, chat model, embedding, or tool entrypoints built on current APIs.
- `libs/langchain`: `langchain-classic`, the legacy compatibility package. Edit this for deprecated imports, backward-compatibility shims, and legacy chain behavior that still needs maintenance.
- `libs/partners`: Provider integrations such as OpenAI and Anthropic. Edit the specific partner package when behavior depends on an external provider SDK or API.
- `libs/standard-tests`: Shared conformance tests for integrations. Edit this when changing the expected contract for chat models, embeddings, tools, retrievers, vector stores, or similar interfaces.
- `libs/text-splitters`: Standalone text chunking utilities. Edit this for document splitting logic and related tests.
- `libs/model-profiles`: The CLI and generated-profile workflow for model capability data. Edit this when changing how profile data is fetched, generated, or refreshed for partner packages.

## Verification by Package

Run verification from the package directory you changed. The commands below are package-local; there is no single root verification command in this checkout.

### Offline unit tests

- `libs/core`: `cd libs/core && make lint && make type && make test`
- `libs/langchain`: `cd libs/langchain && make lint && make type && make test`
- `libs/langchain_v1`: `cd libs/langchain_v1 && make lint && make type && make test_fast`
  Use `make test` only when you need the Docker-backed agent test services; `make test_fast` skips those services and is the safer default for small refactors.
- `libs/standard-tests`: `cd libs/standard-tests && make lint && make type && make test`
- `libs/text-splitters`: `cd libs/text-splitters && make lint && make type && make test`
- `libs/model-profiles`: `cd libs/model-profiles && make lint && make type && make test`
- `libs/partners/openai`: `cd libs/partners/openai && make lint && make type && make test`
  `make test` bootstraps a local `tiktoken_cache/` by downloading tokenizer files before running socket-disabled unit tests.
- `libs/partners/anthropic`: `cd libs/partners/anthropic && make lint && make type && make test`

### Integration tests

- `libs/langchain`: `cd libs/langchain && make integration_tests`
- `libs/langchain_v1`: `cd libs/langchain_v1 && make integration_tests`
- `libs/standard-tests`: `cd libs/standard-tests && make integration_tests`
- `libs/text-splitters`: `cd libs/text-splitters && make integration_tests`
- `libs/model-profiles`: `cd libs/model-profiles && make integration_tests`
- `libs/partners/openai`: `cd libs/partners/openai && make integration_tests`
- `libs/partners/anthropic`: `cd libs/partners/anthropic && make integration_tests`

### Extra prerequisites to call out in review

- `libs/langchain_v1`: full `make test`, `make test_watch`, and `make extended_tests` start Docker services for agent tests via `make start_services`.
- `libs/partners/openai`: unit tests populate `tiktoken_cache/` with tokenizer data before running offline.
- `libs/model-profiles`: `make refresh-profiles` is separate from normal test verification and requires network access.

## Public API Refactor Checklist

Use this checklist before changing exported symbols, compatibility shims, or import paths:

- Inspect the package `__init__.py` first. In this repo, files such as `libs/core/langchain_core/__init__.py`, `libs/langchain_v1/langchain/__init__.py`, and `libs/langchain/langchain_classic/__init__.py` define or expose public entrypoints.
- Check the existing tests and examples that exercise the symbol before moving, renaming, or removing it.
- Do not change function or class signatures unless the change is explicitly approved. Adding, removing, renaming, or reordering parameters should be treated as a breaking change until reviewed.
- Prefer incremental refactors that preserve the current import path and behavior, then add focused tests around the compatibility boundary you touched.
- Stop and ask for additional review before proceeding if the change affects exported imports, deprecation behavior, compatibility layers, generated profile outputs, or anything that could break user code that worked last week.

## LangChain ecosystem

While the LangChain framework can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools when building LLM applications.

- **[Deep Agents](https://github.com/langchain-ai/deepagents)** — Build agents that can plan, use subagents, and leverage file systems for complex tasks
- **[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)** — Build agents that can reliably handle complex tasks with our low-level agent orchestration framework
- **[Integrations](https://docs.langchain.com/oss/python/integrations/providers/overview)** — Chat & embedding models, tools & toolkits, and more
- **[LangSmith](https://www.langchain.com/langsmith)** — Agent evals, observability, and debugging for LLM apps
- **[LangSmith Deployment](https://docs.langchain.com/langsmith/deployments)** — Deploy and scale agents with a purpose-built platform for long-running, stateful workflows

## Why use LangChain?

LangChain helps developers build applications powered by LLMs through a standard interface for models, embeddings, vector stores, and more.

- **Real-time data augmentation** — Easily connect LLMs to diverse data sources and external/internal systems, drawing from LangChain's vast library of integrations with model providers, tools, vector stores, retrievers, and more
- **Model interoperability** — Swap models in and out as your engineering team experiments to find the best choice for your application's needs. As the industry frontier evolves, adapt quickly — LangChain's abstractions keep you moving without losing momentum
- **Rapid prototyping** — Quickly build and iterate on LLM applications with LangChain's modular, component-based architecture. Test different approaches and workflows without rebuilding from scratch, accelerating your development cycle
- **Production-ready features** — Deploy reliable applications with built-in support for monitoring, evaluation, and debugging through integrations like LangSmith. Scale with confidence using battle-tested patterns and best practices
- **Vibrant community and ecosystem** — Leverage a rich ecosystem of integrations, templates, and community-contributed components. Benefit from continuous improvements and stay up-to-date with the latest AI developments through an active open-source community
- **Flexible abstraction layers** — Work at the level of abstraction that suits your needs — from high-level chains for quick starts to low-level components for fine-grained control. LangChain grows with your application's complexity

---

## Documentation

- [docs.langchain.com](https://docs.langchain.com/oss/python/langchain/overview) – Comprehensive documentation, including conceptual overviews and guides
- [reference.langchain.com/python](https://reference.langchain.com/python) – API reference docs for LangChain packages
- [Chat LangChain](https://chat.langchain.com/) – Chat with the LangChain documentation and get answers to your questions

**Discussions**: Visit the [LangChain Forum](https://forum.langchain.com) to connect with the community and share all of your technical questions, ideas, and feedback.

## Additional resources

- [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview) – Learn how to contribute to LangChain projects and find good first issues.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) – Our community guidelines and standards for participation.
- [LangChain Academy](https://academy.langchain.com/) – Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
