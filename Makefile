
build: clean
	python setup.py build_ext -i

docs:
	sh docs.sh

clean:
	@rm -f ateams/*.c*
	@rm -f ateams/*.o
	@rm -f ateams/*.so
	@rm -f ateams/arithmetic/*.html
	@rm -f ateams/arithmetic/*.o
	@rm -f ateams/arithmetic/*.so
	@rm -rf ./build


fp = ./test/output/profiles/metadata/.profiles/profile.json

test: FORCE
	# screen -dm vprof -r
	python test/matrices.py
	# vprof -c mh test/matrices.py --output-file $fp
	# vprof --input-file $fp

FORCE: