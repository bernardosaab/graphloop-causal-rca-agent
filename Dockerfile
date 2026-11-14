# syntax=docker/dockerfile:1.7
#
#   docker build -t graphloop .                       # runtime image
#   docker build --target test .                      # run the whole test suite in a container
#   docker run --rm graphloop run --incident 3 --agent graph+memory
#   docker run --rm -v "$PWD/results:/app/results" graphloop bench --out results

# ---- build: resolve the locked environment with uv ----------------------------------------------
FROM ghcr.io/astral-sh/uv:0.12.5-python3.12-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# ---- test: the same gates as CI, inside the image's environment ----------------------------------
FROM build AS test
RUN uv sync --locked
COPY tests ./tests
RUN uv run pytest -q --cov

# ---- runtime: slim, non-root, no build tooling ---------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
RUN useradd --create-home --uid 10001 graphloop
WORKDIR /app
COPY --from=build --chown=graphloop:graphloop /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER graphloop
ENTRYPOINT ["graphloop"]
CMD ["--help"]
