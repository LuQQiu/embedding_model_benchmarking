#!/bin/bash
set -e

# Deployment helper script for C++ benchmarking on EC2

echo "========================================================================"
echo "C++ Embedding Benchmark - AWS Deployment Helper"
echo "========================================================================"

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "Error: Terraform is not installed"
    echo "Install: brew install terraform (macOS) or download from https://www.terraform.io/"
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed"
    echo "Install: brew install awscli (macOS) or https://aws.amazon.com/cli/"
    exit 1
fi

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials not configured"
    echo "Run: aws configure"
    echo "Or: aws sso login --profile your-profile"
    exit 1
fi

echo "✓ AWS credentials valid"
echo "Account: $(aws sts get-caller-identity --query Account --output text)"
echo "User: $(aws sts get-caller-identity --query Arn --output text)"
echo ""

# Check for SSH key
SSH_KEY="${SSH_PUBLIC_KEY_PATH:-$HOME/.ssh/id_rsa.pub}"
if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH public key not found at $SSH_KEY"
    echo "Generate one with: ssh-keygen -t rsa -b 4096"
    exit 1
fi

echo "✓ SSH key found: $SSH_KEY"
echo ""

# Change to terraform directory
cd "$(dirname "$0")/terraform"

# Parse command
COMMAND="${1:-help}"

case "$COMMAND" in
    init)
        echo "Initializing Terraform..."
        terraform init
        echo ""
        echo "✓ Terraform initialized"
        ;;

    plan)
        echo "Planning deployment..."
        terraform plan
        ;;

    apply|deploy)
        echo "Deploying C++ benchmark instance..."
        echo ""
        echo "This will create:"
        echo "  - EC2 instance (c7i.8xlarge, ~\$1.36/hour)"
        echo "  - Elastic IP"
        echo "  - Security group"
        echo ""
        read -p "Continue? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo "Deployment cancelled"
            exit 0
        fi

        terraform apply

        echo ""
        echo "========================================================================"
        echo "Deployment complete!"
        echo "========================================================================"
        echo ""
        terraform output quick_start
        ;;

    ssh)
        echo "Connecting to instance..."
        IP=$(terraform output -raw public_ip 2>/dev/null)
        if [ -z "$IP" ]; then
            echo "Error: No instance deployed. Run: $0 deploy"
            exit 1
        fi
        ssh -i ~/.ssh/id_rsa ubuntu@"$IP"
        ;;

    status)
        echo "Instance status:"
        INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null)
        if [ -z "$INSTANCE_ID" ]; then
            echo "No instance deployed"
            exit 0
        fi

        aws ec2 describe-instances \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].[InstanceId,State.Name,InstanceType,PublicIpAddress]' \
            --output table

        echo ""
        echo "Setup log (last 20 lines):"
        IP=$(terraform output -raw public_ip 2>/dev/null)
        ssh -i ~/.ssh/id_rsa ubuntu@"$IP" "tail -20 /var/log/user-data.log" 2>/dev/null || echo "Instance not ready yet"
        ;;

    stop)
        echo "Stopping instance (preserves data, stops compute charges)..."
        INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null)
        if [ -z "$INSTANCE_ID" ]; then
            echo "No instance to stop"
            exit 0
        fi
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
        echo "✓ Instance stopping"
        ;;

    start)
        echo "Starting instance..."
        INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null)
        if [ -z "$INSTANCE_ID" ]; then
            echo "No instance to start"
            exit 0
        fi
        aws ec2 start-instances --instance-ids "$INSTANCE_ID"
        echo "✓ Instance starting"
        ;;

    destroy)
        echo "Destroying infrastructure..."
        echo ""
        echo "WARNING: This will delete:"
        echo "  - EC2 instance and all data"
        echo "  - Elastic IP"
        echo "  - Security group"
        echo ""
        read -p "Are you sure? Type 'yes' to confirm: " confirm
        if [ "$confirm" != "yes" ]; then
            echo "Destruction cancelled"
            exit 0
        fi

        terraform destroy

        echo ""
        echo "✓ Infrastructure destroyed"
        ;;

    cost)
        echo "Cost estimation:"
        echo ""
        echo "c7i.8xlarge: \$1.36/hour"
        echo "  - Per day (24h): \$32.64"
        echo "  - Per week: \$228.48"
        echo "  - Per month: \$979.20"
        echo ""
        echo "Storage (100GB gp3): \$8/month"
        echo "Elastic IP (attached): Free"
        echo "Data transfer: Variable"
        echo ""
        echo "Estimated total: ~\$1.40/hour when running"
        echo ""
        echo "Cost-saving tips:"
        echo "  - Stop instance when not in use (preserves data)"
        echo "  - Use c7i.4xlarge for smaller workloads (\$0.68/hour)"
        echo "  - Destroy when project complete"
        ;;

    help|*)
        cat << EOF
Usage: $0 <command>

Commands:
  init       Initialize Terraform (first time setup)
  plan       Preview changes before deployment
  deploy     Deploy EC2 instance for C++ benchmarking
  ssh        SSH into the deployed instance
  status     Show instance status and setup progress
  stop       Stop instance (preserves data, stops billing)
  start      Start stopped instance
  destroy    Destroy all infrastructure (deletes everything!)
  cost       Show cost estimation
  help       Show this help message

Examples:
  $0 init              # First time setup
  $0 deploy            # Deploy instance
  $0 ssh               # Connect to instance
  $0 status            # Check instance status
  $0 stop              # Stop when not in use
  $0 destroy           # Clean up when done

For more information, see terraform/README.md
EOF
        ;;
esac
