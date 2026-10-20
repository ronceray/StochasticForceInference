# docs/Makefile

SPHINXOPTS  ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = source
BUILDDIR    = build

.PHONY: help clean html apidoc

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@rm -f "$(SOURCEDIR)"/api/*.rst
	@rm -rf $(SOURCEDIR)/generated

html:
	@JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_ALLOCATOR=platform $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@sphinx-apidoc -o "$(SOURCEDIR)/api" -f -e ../SFI \
		../SFI/statefunc/OLD_stateexpr.py \
		../SFI/statefunc/nodes/OLD_interactions.py \
		"../SFI/statefunc/*conflicted*.py"

