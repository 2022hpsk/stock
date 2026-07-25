.PHONY: help install fmt lint type test check arch secrets clean ui ui-dev ui-build

help:
	@echo "install  安装依赖（uv sync）"
	@echo "ui       启动本地可视化界面 http://127.0.0.1:8686"
	@echo "ui-dev   前端开发模式（Vite 热更新，需 Node）"
	@echo "ui-build 构建前端产物到 src/quantstock/web/static/"
	@echo "fmt      格式化代码"
	@echo "lint     ruff 静态检查"
	@echo "type     mypy 严格类型检查"
	@echo "arch     import-linter 分层依赖检查"
	@echo "secrets  密钥泄漏扫描"
	@echo "test     pytest + 覆盖率"
	@echo "check    提交前必跑：fmt-check + lint + type + arch + secrets + test"

install:
	uv sync --all-extras --all-groups

ui:
	uv run quantstock ui --open

ui-dev:
	cd frontend && npm install && npm run dev

ui-build:
	cd frontend && npm install && npm run build

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

type:
	uv run mypy

arch:
	uv run lint-imports

secrets:
	uv run detect-secrets scan --baseline .secrets.baseline

test:
	uv run pytest

check: lint type arch test
	@echo "✅ 全部检查通过"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
