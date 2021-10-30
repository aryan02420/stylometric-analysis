.PHONY: all corpora test
.ONESHELL:
SHELL := /bin/bash
corpora := aurthur_conan_doyle charles_dickens h_g_wells jane_austen jonathan_swift

all:
	$(MAKE) clean
	$(MAKE) corpora
	$(MAKE) corpora_split

clean:
	rm -f corpora/**
	rm -rf corpora_split/**

corpora:	$(addprefix c_,$(corpora))

corpora_split:	$(addprefix s_,$(corpora))

c_%:
	export startp="\*\*\*\s*START\sOF"
	export endp="\*\*\*\s*END\sOF"
	echo > corpora/$*
	find data_sources/$* -type f | \
	xargs -i bash -c "sed -e '/$$startp/,/$$endp/p' -n {} | tail -n+2 | head -n-1 >> corpora/$*"

s_%:
	mkdir corpora_split/$* | true
	export startp="\*\*\*\s*START\sOF"
	export endp="\*\*\*\s*END\sOF"
	ls data_sources/$* | xargs -i -n 1 bash -c "sed -e '/$$startp/,/$$endp/p' -n data_sources/$*/{} | tail -n+2 | head -n-1 | split -d -a 4 -l 700 - corpora_split/$*/{}."

download:
	python utils/download.py

pycache:
	echo y | rm -rf lib/__pycache__/

prefix_%:
	echo $*