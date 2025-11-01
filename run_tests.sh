#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}ERROR: python3 command not found!${NC}"
        echo "Please install Python 3 to run this script."
        exit 1
    fi
    echo "Found Python: $(python3 --version)"
}

ensure_python_packages() {
    PACKAGES=("transformers" "torch")
    MISSING_PACKAGES=()

    for pkg in "${PACKAGES[@]}"; do
        pkg_import=${pkg//-/_}
        if ! python3 -c "import $pkg_import" &> /dev/null; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done

    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo -e "${YELLOW}Installing required Python packages: ${MISSING_PACKAGES[*]}${NC}"
        pip install "${MISSING_PACKAGES[@]}"
    fi
}

generate_test_data() {
    echo -e "${YELLOW}Generating test data...${NC}"

    check_python
    ensure_python_packages

    pushd python > /dev/null
    python3 generate_test_data.py
    popd > /dev/null

    echo -e "${GREEN}Test data generated successfully!${NC}"
}

invoke_dotnet_checks() {
    echo -e "${YELLOW}Running .NET checks and tests...${NC}"

    if ! command -v dotnet &> /dev/null; then
        echo -e "${RED}ERROR: dotnet command not found!${NC}"
        echo "Please install .NET SDK to run .NET tests."
        exit 1
    fi

    echo "Found .NET: $(dotnet --version)"

    echo -e "${YELLOW}Verifying code formatting...${NC}"
    if ! dotnet format Yuniko.Software.Qwen3Tokenizer.slnx --verify-no-changes --verbosity diagnostic; then
        echo -e "${RED}ERROR: Code formatting issues detected!${NC}"
        echo "Run 'dotnet format' to fix formatting issues."
        exit 1
    fi
    echo -e "${GREEN}Code formatting verified!${NC}"

    dotnet restore Yuniko.Software.Qwen3Tokenizer.slnx
    dotnet build Yuniko.Software.Qwen3Tokenizer.slnx --no-restore --configuration Release
    dotnet test Yuniko.Software.Qwen3Tokenizer.slnx --no-build --configuration Release --verbosity normal --collect:"XPlat Code Coverage"

    echo -e "${GREEN}All .NET checks passed!${NC}"
}

echo -e "${YELLOW}Starting Qwen3 Tokenizer test suite${NC}"
echo ""

generate_test_data
echo ""

invoke_dotnet_checks
echo ""

echo -e "${GREEN}All checks and tests passed successfully!${NC}"
exit 0
