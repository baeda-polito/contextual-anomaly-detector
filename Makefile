# Makefile for python code
# https://gist.github.com/MarkWarneke/2e26d7caef237042e9374ebf564517ad

define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

VENV := source .venv/bin/activate


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
	docker run cmp