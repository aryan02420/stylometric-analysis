.PHONY: all corpus test
SHELL := /bin/bash
corpora := aurthur_conan_doyle charles_dickens h_g_wells jane_austen jonathan_swift

all:
	$(MAKE) clean
	$(MAKE) corpora

clean:
	rm -f corpora/**

corpora:	$(corpora)

$(corpora):
	export startp="START\sOF\sTHE\sPROJECT\sGUTENBERG\sEBOOK" && \
	export endp="END\sOF\sTHE\sPROJECT\sGUTENBERG\sEBOOK" && \
	echo > corpora/$@ && \
	find data_sources/$@ -type f | \
	xargs -i bash -c "sed -e '/$$startp/,/$$endp/p' -n {} | tail -n+2 | head -n-1 | awk NF >> corpora/$@"

download:
	python utils/download.py

pycache:
	echo y | rm -rf lib/__pycache__/*