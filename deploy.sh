#!/bin/bash
set -e

# Automated AWS Deployment Script for Embedding Model Benchmarking
# This script guides you through the deployment process

echo "======================================================================="
echo "Embedding Model Benchmark - AWS Deployment"
echo "======================================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}✗ AWS CLI not found${NC}"
    echo "  Install: https://aws.amazon.com/cli/"
    exit 1
fi
echo -e "${GREEN}✓ AWS CLI found: $(aws --version)${NC}"

# Check Terraform
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}✗ Terraform not found${NC}"
    echo "  Install: https://developer.hashicorp.com/terraform/downloads"
    exit 1
fi
echo -e "${GREEN}✓ Terraform found: $(terraform --version | head -n1)${NC}"

# Check SSH key
if [ ! -f ~/.ssh/id_rsa.pub ]; then
    echo -e "${YELLOW}⚠ SSH public key not found at ~/.ssh/id_rsa.pub${NC}"
    echo "  Generate one with: ssh-keygen -t rsa -b 4096"
    read -p "Do you have an SSH key at a different location? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check AWS credentials
echo ""
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}✗ AWS credentials not configured${NC}"
    echo ""
    echo "Run: aws configure"
    echo "You'll need:"
    echo "  - AWS Access Key ID"
    echo "  - AWS Secret Access Key"
    echo "  - Default region (us-east-1 recommended)"
    exit 1
fi

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_USER=$(aws sts get-caller-identity --query Arn --output text)
echo -e "${GREEN}✓ Authenticated as: $AWS_USER${NC}"
echo -e "${GREEN}  Account: $AWS_ACCOUNT${NC}"

# Get configuration
echo ""
echo "======================================================================="
echo "Configuration"
echo "======================================================================="

# Get user's IP for security group
echo ""
echo "Getting your IP address for security group..."
MY_IP=$(curl -s ifconfig.me)
echo -e "${GREEN}Your IP: $MY_IP${NC}"

# Prompt for customization
echo ""
read -p "Enter a unique project prefix (e.g., yourname-emb): " PROJECT_PREFIX
PROJECT_PREFIX=${PROJECT_PREFIX:-lu-emb-bench}

read -p "Enter your name for resource tagging: " OWNER
OWNER=${OWNER:-admin}

echo ""
echo "Region selection:"
echo "  1) us-east-1 (N. Virginia) - Recommended, lowest cost (~$1.36/hr)"
echo "  2) us-west-2 (Oregon) - West Coast, slightly higher cost (~$1.44/hr)"
read -p "Choose region (1 or 2): " REGION_CHOICE

if [ "$REGION_CHOICE" = "2" ]; then
    AWS_REGION="us-west-2"
else
    AWS_REGION="us-east-1"
fi

echo ""
echo "Configuration summary:"
echo "  Project prefix: $PROJECT_PREFIX"
echo "  Owner: $OWNER"
echo "  Region: $AWS_REGION"
echo "  Allowed SSH IP: $MY_IP/32"
echo "  Instance type: c7i.8xlarge (32 vCPU, 64GB RAM)"
echo ""
read -p "Proceed with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Create terraform.tfvars
echo ""
echo "Creating terraform configuration..."
cd terraform

cat > terraform.tfvars <<EOF
# Auto-generated configuration
project_prefix  = "$PROJECT_PREFIX"
owner           = "$OWNER"
aws_region      = "$AWS_REGION"
allowed_ssh_cidr = ["$MY_IP/32"]
instance_type   = "c7i.8xlarge"
root_volume_size = 100
environment     = "benchmark"
EOF

echo -e "${GREEN}✓ Configuration saved to terraform/terraform.tfvars${NC}"

# Initialize Terraform
echo ""
echo "======================================================================="
echo "Initializing Terraform"
echo "======================================================================="
terraform init

# Plan
echo ""
echo "======================================================================="
echo "Planning deployment"
echo "======================================================================="
terraform plan -out=tfplan

# Apply
echo ""
echo "======================================================================="
echo "Deploying infrastructure"
echo "======================================================================="
echo ""
echo -e "${YELLOW}This will create AWS resources and incur costs!${NC}"
echo "  - EC2 c7i.8xlarge: ~$1.36-1.44/hour"
echo "  - EBS 100GB gp3: ~$8/month"
echo ""
read -p "Continue with deployment? (yes/no) " -r
if [[ ! $REPLY = "yes" ]]; then
    echo "Deployment cancelled"
    rm tfplan
    exit 0
fi

terraform apply tfplan
rm tfplan

# Get outputs
echo ""
echo "======================================================================="
echo "Deployment Complete!"
echo "======================================================================="

INSTANCE_IP=$(terraform output -raw instance_public_ip)
INSTANCE_ID=$(terraform output -raw instance_id)

echo ""
echo -e "${GREEN}✓ Instance deployed successfully${NC}"
echo ""
echo "Instance details:"
echo "  IP Address: $INSTANCE_IP"
echo "  Instance ID: $INSTANCE_ID"
echo "  Region: $AWS_REGION"
echo "  Instance Type: c7i.8xlarge"
echo ""
echo "SSH command:"
echo "  ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP"
echo ""

# Save connection info
cd ..
cat > instance_info.sh <<EOF
# Instance connection information
export INSTANCE_IP=$INSTANCE_IP
export INSTANCE_ID=$INSTANCE_ID
export AWS_REGION=$AWS_REGION
EOF

echo -e "${GREEN}✓ Connection info saved to instance_info.sh${NC}"
echo ""

# Wait for instance to be ready
echo "======================================================================="
echo "Waiting for instance to be ready..."
echo "======================================================================="
echo "This may take 2-3 minutes..."

sleep 30  # Initial wait

MAX_ATTEMPTS=20
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    STATUS=$(aws ec2 describe-instance-status \
        --instance-ids $INSTANCE_ID \
        --region $AWS_REGION \
        --query 'InstanceStatuses[0].InstanceStatus.Status' \
        --output text 2>/dev/null || echo "initializing")

    if [ "$STATUS" = "ok" ]; then
        echo -e "${GREEN}✓ Instance is ready!${NC}"
        break
    fi

    echo "  Status: $STATUS (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)"
    sleep 10
    ATTEMPT=$((ATTEMPT+1))
done

echo ""
echo "======================================================================="
echo "Next Steps"
echo "======================================================================="
echo ""
echo "1. Connect to instance:"
echo "   ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP"
echo ""
echo "2. Clone repository and setup:"
echo "   git clone <your-repo-url>"
echo "   cd embedding_model_benchmarking"
echo "   bash scripts/setup.sh"
echo ""
echo "3. Download and convert models:"
echo "   python3 scripts/download_model.py --model embeddinggemma-300m"
echo "   python3 scripts/convert_to_onnx.py --model embeddinggemma-300m"
echo "   python3 scripts/convert_to_openvino.py --model embeddinggemma-300m"
echo ""
echo "4. Build Docker images:"
echo "   docker-compose build"
echo ""
echo "5. Run benchmarks:"
echo "   python3 orchestrator/runner.py --model embeddinggemma-300m"
echo ""
echo "======================================================================="
echo -e "${YELLOW}IMPORTANT: Remember to destroy resources when done!${NC}"
echo "   cd terraform && terraform destroy"
echo "======================================================================="
echo ""

# Offer to SSH
read -p "Connect to instance now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP
fi
