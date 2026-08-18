[windows]
set shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# List the available recipes.
default:
    @just --list

# Download the upstream Ashaar dataset.
download output="data/ashaar_dataset.parquet" *args:
    uv run python download_ashaar.py --output {{ output }} {{ args }}

# Run the offline test suite.
test *args:
    uv run python -m unittest discover -s tests -v {{ args }}

# Measure and certify the three configured endpoints.
benchmark task="poem-generation" *args:
    uv run ai-poet-benchmark-endpoints --task {{ task }} --insecure {{ args }}

# Run the pilot using the task's certified capacity report.
pilot task="poem-generation" *args:
    uv run ai-poet-pilot-sft --task {{ task }} --capacity-report {{ if task == "poem-generation" { "data/gemma_capacity" } else { "data/gemma_capacity_" + replace(task, "-", "_") } }}/endpoint_capacity.json --insecure {{ args }}

# Run or resume safeguarded three-endpoint generation.
generate task="poem-generation" *args:
    uv run ai-poet-generate-sft --task {{ task }} --capacity-report {{ if task == "poem-generation" { "data/gemma_capacity" } else { "data/gemma_capacity_" + replace(task, "-", "_") } }}/endpoint_capacity.json --pilot-report {{ if task == "poem-generation" { "data/ashaar_sft" } else { "data/ashaar_" + replace(task, "poem-", "") + "_sft" } }}/pilot_report.json --pilot-review {{ if task == "poem-generation" { "data/ashaar_sft" } else { "data/ashaar_" + replace(task, "poem-", "") + "_sft" } }}/pilot_review.json --insecure {{ args }}

# Run or resume generation against one endpoint.
generate-single task="poem-generation" concurrency="32" *args:
    uv run ai-poet-generate-sft --task {{ task }} --concurrency {{ concurrency }} --insecure {{ args }}

# Run generation against three endpoints without benchmark or pilot gates.
generate-unsafe task="poem-generation" *args:
    uv run ai-poet-generate-sft --task {{ task }} --skip-pilot-review --insecure {{ args }}

# Show every generator option.
generate-help:
    uv run ai-poet-generate-sft --help
