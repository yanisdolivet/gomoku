#!/bin/bash
# scripts/install-hooks.sh
# This script downloads and installs git hooks for this repository

# 1. Hook commit-msg : Checks the regex [TYPE]
cat <<EOT > .git/hooks/commit-msg
#!/bin/bash
INPUT_FILE=\$1
COMMIT_MSG=\`head -n1 \$INPUT_FILE\`
PATTERN="^\[(ADD|FIX|UPDATE|REMOVE|DOC|REFACTOR|TEST|CI|CONFIG|MERGE|REBASE)\] .+"

if ! [[ "\$COMMIT_MSG" =~ \$PATTERN ]]; then
  echo "Error : Invalid commit message."
  echo "Necessary format : [TYPE] Message in English"
  echo "Example : [ADD] Create parsing logic"
  exit 1
fi
EOT

# 2. Hook pre-commit : Checks that it compiles before commit
cat <<EOT > .git/hooks/pre-commit
#!/bin/bash
echo "Checking compilation..."
make
RET=\$?
make fclean
if [ \$RET -ne 0 ]; then
    echo "Compilation failed. Commit cancelled."
    exit 1
fi
EOT

chmod +x .git/hooks/commit-msg
chmod +x .git/hooks/pre-commit
echo "Hooks Git successfully installed."