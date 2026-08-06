ENV   := grad
CONDA := conda

.PHONY: install uninstall

install:
	$(CONDA) env create -f environment.yml

uninstall:
	$(CONDA) env remove -n $(ENV) --yes
