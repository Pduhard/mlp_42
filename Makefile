
.PHONY = help setup test run

.DEFAULT_GOAL = help
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
PYTHON=~/.brew/Cellar/python@3.8/3.8.8_1/bin/python3.8
else
PYTHON=python
endif

LIB=protodeep

help:
	@echo ---------------HELP-----------------
	@echo To setup the project and build packages type make setup
	@echo ------------------------------------

setup:
	make setup -C $(LIB)
