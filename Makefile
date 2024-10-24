# Makefile to clean Python cache files and remove the data folder

# Specify the folders to clean
APP_FOLDER = app
DATA_FOLDER = data

# Define the clean command
.PHONY: clean
clean:
	@echo "Removing Python cache files from $(APP_FOLDER)..."
	find $(APP_FOLDER) -name "__pycache__" -type d -exec rm -r {} + || true
	find $(APP_FOLDER) -name "*.pyc" -type f -delete || true
	find $(APP_FOLDER) -name "*.pyo" -type f -delete || true
	@echo "Cache files removed."

# Define the remove_data command
.PHONY: remove_data
remove_data:
	@echo "Removing data folder..."
	rm -rf $(DATA_FOLDER)
	@echo "$(DATA_FOLDER) folder removed."

# Define a clean_all rule to run both clean and remove_data
.PHONY: clean_all
clean_all: clean remove_data
	@echo "All cleaning tasks completed."

# Optional: Add an "all" rule to run everything
.PHONY: all
all: clean_all
	@echo "All tasks completed."
