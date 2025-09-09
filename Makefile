
SHELL := /bin/zsh
DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

###############
#### BUILD ####
###############
clean:
	@rm -rf ./build

quick: FORCE
	@python setup.py build_ext --inplace


build: clean PHATMethods LinBoxMethods
	@python setup.py build_ext --inplace > build.log 2>&1 


PHATMethods: common
	@sudo cp -r ateams/arithmetic/include/PHAT /usr/local/include/phat
	@sudo clang++ -shared -fPIC -std=c++17 -o /usr/local/lib/libPHATMethods.so ateams/arithmetic/PHATMethods.cpp -v -O3
	@sudo cp -r ateams/arithmetic/PHATMethods.h /usr/local/include/ATEAMS/


LinBoxMethods: common
	@sudo cp -r ateams/arithmetic/LinBoxMethods.h /usr/local/include/ATEAMS/
	@sudo clang++ `pkg-config --libs linbox` -shared -fPIC -std=c++17 -o /usr/local/lib/libLinBoxMethods.so ateams/arithmetic/LinBoxMethods.cpp -v -O3


common:
	@sudo cp -r ateams/common.h /usr/local/include/ATEAMS/




#################
#### PROFILE ####
#################
Glauber: FORCE
	@cd test && ./profile.models.Glauber.sh 19 22 4
	@cd test && ./profile.models.Glauber.sh 999 1002 2

SwendsenWang: FORCE
	@cd test && ./profile.models.SW.sh 4 7 4
	@cd test && ./profile.models.SW.sh 499 502 2

Nienhuis: FORCE
	@cd test && ./profile.models.NH.sh 49 52 2
	@cd test && ./profile.models.NH.sh 19 22 3

InvadedCluster: FORCE
	@cd test && ./profile.models.IC.sh 7 8 4
	@cd test && ./profile.models.IC.sh 19 22 2

Bernoulli: FORCE
	@cd test && ./profile.models.Bernoulli.sh 4 7 4
	@cd test && ./profile.models.Bernoulli.sh 19 22 2

profile: Glauber SwendsenWang Nienhuis InvadedCluster Bernoulli

test: FORCE
	@cd test && sudo clang++ -L/usr/local/include/ATEAMS/ `pkg-config --libs linbox` -lLinBoxMethods test.arithmetic.persistence.cpp -o test.arithmetic.persistence

gauntlet: FORCE test profile

# Kills all the profiles (in case something goes wrong, or is taking too long).
killall: FORCE
	@ps aux | grep -e python -e make -e profile.models | awk '{print $$2}' | xargs kill




#######################
#### DOCUMENTATION ####
#######################
tables: FORCE
	@echo "This will hang if you aren't logged in as anthony on Pangolin."
	@cd test/stats && ./stats.sync.sh && ./stats.tables.sh

docs: FORCE quick
	@pdoc ateams --force --html --template-dir docs/templates --output-dir=docs
	@rsync -a docs/ateams/ docs/
	@rm -rf docs/ateams
	@open "file://$(DIR)docs/index.html"

refs: FORCE
	@pandoc -t markdown_strict --citeproc _refs.md -o refs.md

contribute: build profile docs





#################
#### INSTALL ####
#################
install: dependencies _install profile

dependencies: FORCE
	@pip install -r requirements.txt
	@sudo cp -r ateams/common.h /usr/local/include/ATEAMS/

_install: FORCE build
	python setup.py develop

FORCE:
