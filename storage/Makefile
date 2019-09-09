.PHONY: make-WinSCP-doc

make-WinSCP-doc: submodule-update 
	cp ../NIRD_doc/WinSCP.md .
	cp -r ../NIRD_doc/images .

.PHONY: submodule-update
submodule-update:
	git submodule init
	git submodule update
