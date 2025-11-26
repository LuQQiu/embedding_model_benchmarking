#!/bin/bash
# Helper script for HuggingFace authentication

echo "========================================="
echo "HuggingFace Authentication"
echo "========================================="
echo ""
echo "This script will help you authenticate with HuggingFace"
echo "to download gated models like google/embeddinggemma-300m"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found"
    echo ""
    echo "Please run the setup script first:"
    echo "  bash scripts/setup.sh"
    echo ""
    echo "Or add ~/.local/bin to your PATH:"
    echo "  export PATH=\$PATH:~/.local/bin"
    exit 1
fi

echo "Step 1: Get your HuggingFace token"
echo "  1. Go to: https://huggingface.co/settings/tokens"
echo "  2. Click 'New token' or copy an existing token"
echo "  3. Token type: Read or Write (Read is sufficient)"
echo ""

echo "Step 2: Accept model terms (for gated models)"
echo "  For EmbeddingGemma:"
echo "  - Visit: https://huggingface.co/google/embeddinggemma-300m"
echo "  - Click 'Agree and access repository'"
echo ""

echo "Step 3: Login"
echo "  The huggingface-cli login command will prompt you for your token"
echo ""

read -p "Ready to login? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    huggingface-cli login

    echo ""
    echo "========================================="
    echo "Authentication complete!"
    echo "========================================="
    echo ""
    echo "You can now download gated models:"
    echo "  python3 scripts/download_model.py --model embeddinggemma-300m"
    echo ""
else
    echo ""
    echo "Authentication cancelled."
    echo "You can run this script again anytime: bash scripts/huggingface_login.sh"
    echo ""
fi
