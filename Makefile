#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = cgr-case-study
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	conda env update --name cgr-frag --file envs/cgr-frag.yml --prune
	conda env update --name ml-env --file envs/ml-env.yml --prune
	conda env update --name dl-env --file envs/dl-env.yml --prune

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 rcr_benchmarking
	isort --check --diff --profile black rcr_benchmarking
	black --check --config pyproject.toml rcr_benchmarking

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml rcr_benchmarking




## Set up python interpreter environment
.PHONY: create_environments
create_environment:
	conda env create --name cgr-frag -f envs/cgr-frag.yml
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

	conda env create --name ml-env -f envs/ml-env.yml
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

	conda env create --name dl-env -f envs/dl-env.yml
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	chmod +x ./make_jacs_dataset_and_frags.sh
	./make_jacs_dataset_and_frags.sh


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
