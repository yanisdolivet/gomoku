##
## EPITECH PROJECT, 2025
## Delivery
## File description:
## Makefile
##

## ------------------------------------ ##
##              VARIABLES               ##

CC                  := g++
CFLAGS              := -Wall -Wextra -march=native -I./include \
						-I./include/Core -I./include/Engine \
						-I./include/Neural -I./include/Protocol \
						-I./include/Utils
DFLAGS              := -g3
TFLAGS				:= -lcriterion --coverage

EXECUTABLE          := pbrain-gomoku-ai
EXEC_ARGS			:= -help
TEST_EXECUTABLE     := unit_tests

OBJDIR              := obj
SRCDIR              := src
TESTDIR             := tests


SOURCES_FILES       := $(shell find $(SRCDIR) -name '*.cpp')
OBJECTS_FILES       := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,\
						$(SOURCES_FILES))
TEST_SOURCES        := $(filter-out $(SRCDIR)/main.cpp, $(SOURCES_FILES))
TEST_OBJECTS        := $(shell find $(TESTDIR) -name '*.cpp')

RESET               := \033[0m
GREEN               := \033[32m
BLUE                := \033[34m
CYAN                := \033[36m
RED                 := \033[31m

DEBUG ?= 1
ifeq ($(DEBUG), 1)
	CFLAGS += $(DFLAGS)
endif

## ------------------------------------ ##
##                RULES                 ##

all: $(EXECUTABLE)
	@echo -e "$(GREEN)[✔] Compilation complete.$(RESET)"

$(EXECUTABLE): $(OBJECTS_FILES)
	@mkdir -p $(@D)
	@echo -e "$(CYAN)[➜] Linking$(RESET)"
	@$(CC) $(CFLAGS) $^ -o $@
	@echo -e "$(GREEN)[✔] Executable created: $(EXECUTABLE)$(RESET)"

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(@D)
	@echo -e "$(BLUE)[~] Compiling: $<$(RESET)"
	@$(CC) $(CFLAGS) -c $< -o $@

clean:
	@rm -rf $(OBJDIR)
	@rm -f $(TEST_EXECUTABLE)
	@rm -f *.gcno
	@rm -f *.gcda
	@rm -f vgcore.*
	@echo -e "$(RED)[✘] Objects and coverage files removed.$(RESET)"

fclean: clean
	@rm -f $(EXECUTABLE) $(TEST_EXECUTABLE)
	@echo -e "$(RED)[✘] Executables removed.$(RESET)"

re: fclean
	@$(MAKE) all --no-print-directory

## ------------------------------------ ##
##              UNIT TESTS               ##

tests_run:
	@echo -e "$(CYAN)[➜] Linking tests$(RESET)"
	@$(CC) $(CFLAGS) $(TEST_SOURCES) $(TEST_OBJECTS) \
	$(TFLAGS) -o $(TEST_EXECUTABLE)
	@echo -e "$(GREEN)[✔] Unit tests executable created: \
	$(TEST_EXECUTABLE)$(RESET)"
	@echo -e "$(CYAN)[➜] Running unit tests$(RESET)"
	@./$(TEST_EXECUTABLE)


coverage: tests_run
	@echo -e "$(CYAN)[➜] Generating code coverage report$(RESET)"
	@gcovr --exclude $(TESTDIR)

.PHONY: all clean fclean re tests_run coverage
