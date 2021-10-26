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
	echo > corpora/$@ && \
	find data_sources/$@ -type f | \
	xargs -i bash -c "sed -e '/START\sOF\sTHE\sPROJECT\sGUTENBERG\sEBOOK/,/END\sOF\sTHE\sPROJECT\sGUTENBERG\sEBOOK/p' -n {} | tail -n+2 | head -n-1 >> corpora/$@"

