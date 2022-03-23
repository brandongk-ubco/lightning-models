mypy lightningmodels
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count  --max-complexity=10 --statistics
yapf --diff -r lightningmodels