# gradients environment setup
ENV := grad

.PHONY: install uninstall

# create the conda env and set GRAD_ROOT_DIR to this repo
install:
	conda env create -f environment.yml
	conda env config vars set -n $(ENV) GRAD_ROOT_DIR="$(CURDIR)"

# remove the conda env (must not be active; conda refuses to remove the current env)
uninstall:
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV)" ]; then \
		echo "Env '$(ENV)' is active. Run 'conda deactivate' first, then 'make uninstall'."; \
	else \
		conda env remove -n $(ENV) -y; \
	fi
