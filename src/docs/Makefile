# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DOCS_root     = docs
DOCS	      = ../../docs
this_dir      = src/docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

github:
	@rm -rf "$(BUILDDIR)"
	@rm -rf "$(DOCS)/"*
	@cd "../../"
	@git checkout gh-pages
	@rm -rf "$(DOCS_root)/"*
	@git checkout master
	@cd "$(this_dir)
	@make html
	@mv -v "$(BUILDDIR)/html/"* "$(DOCS)"
	@touch "$(DOCS)/.nojekyll"
	@git checkout gh-pages
	@git add "$(DOCS)"
	@git commit -m "update doc"
	@git push
