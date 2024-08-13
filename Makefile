# Makefile for python code
# https://gist.github.com/MarkWarneke/2e26d7caef237042e9374ebf564517ad

define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef
VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

help:
	@echo 'The following commands can be used.'
	@echo ''
	$(call find.functions)


clean: ## Remove build and cache files
clean:
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf build
	rm -rf dist
	rm -rf .pytest_cache
	# Remove all pycache
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

docker-build: ## Build docker image
docker-build:
	docker build -t cmp .

docker-run: ## Build docker image
docker-run:
	docker run cmp data/data.csv Total_Power results/reports/report.html




.PHONY: rm-git-cache
rm-git-cache:
	@echo "Removing git cached files"
	git add .
	git rm -r --cached .
	git add .

.PHONY: setup
setup:
	@if [ ! -d "${VENV}" ]; then \
		echo "Creating venv"; \
		python3 -m venv ${VENV}; \
	fi
	echo "Venv already exists"; \
	source ${VENV}/bin/activate; \
	pip install --upgrade pip; \
	pip install poetry; \
	poetry config virtualenvs.create false; \
	poetry install;