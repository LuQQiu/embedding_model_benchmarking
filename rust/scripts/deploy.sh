#!/bin/bash
#
# Deploy Rust benchmarking infrastructure
#
# Usage:
#   ./scripts/deploy.sh
#

set -e

echo "========================================================================"
echo "Rust Benchmark Deployment"
echo "========================================================================"

# Check Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "Error: Terraform not installed"
    echo "Install from: https://www.terraform.io/downloads"
    exit 1
fi

# Check AWS credentials
if [ -z "$AWS_ACCESS_KEY_ID" ] && [ ! -f ~/.aws/credentials ]; then
    echo "Error: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

cd terraform

echo ""
echo "1. Initializing Terraform..."
terraform init

echo ""
echo "2. Planning deployment..."
terraform plan

echo ""
read -p "Deploy infrastructure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Deployment cancelled"
    exit 0
fi

echo ""
echo "3. Applying Terraform configuration..."
terraform apply -auto-approve

echo ""
echo "4. Getting instance IP..."
INSTANCE_IP=$(terraform output -raw instance_public_ip)

echo ""
echo "========================================================================"
echo "Deployment complete!"
echo "========================================================================"
echo "Instance IP: $INSTANCE_IP"
echo ""
echo "SSH command:"
echo "  ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP"
echo ""
echo "Next steps:"
echo "  1. SSH to instance"
echo "  2. Clone repository"
echo "  3. Copy models and config"
echo "  4. Build Docker containers"
echo "  5. Run benchmarks"
echo ""
echo "See README.md for detailed instructions"
echo "========================================================================"
