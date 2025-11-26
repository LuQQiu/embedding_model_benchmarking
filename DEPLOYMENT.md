# AWS Deployment Guide

Complete step-by-step guide for deploying the embedding model benchmark infrastructure.

## Prerequisites

1. **AWS CLI installed**
   ```bash
   # Check if installed
   aws --version

   # If not installed:
   # macOS: brew install awscli
   # Linux: sudo apt install awscli
   # Or: pip install awscli
   ```

2. **SSH key pair**
   ```bash
   # Check if you have SSH keys
   ls -la ~/.ssh/id_rsa.pub

   # If not, generate one:
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```

3. **Terraform installed**
   ```bash
   # Check if installed
   terraform --version

   # If not installed:
   # macOS: brew install terraform
   # Linux: https://developer.hashicorp.com/terraform/downloads
   ```

## Region Selection

### Recommended Regions

**Primary: us-east-1 (N. Virginia)**
- ✅ Best c7i instance availability
- ✅ Typically lowest pricing (~$1.36/hour)
- ✅ Most mature region
- ✅ Best for testing/development
- ⚠️ Higher network latency for West Coast users

**Alternative: us-west-2 (Oregon)**
- ✅ Excellent c7i availability
- ✅ Lower latency for West Coast
- ✅ Very reliable infrastructure
- ⚠️ Slightly higher pricing (~$1.44/hour, ~6% more)
- ✅ Good choice if you're on West Coast

**Comparison:**

| Feature | us-east-1 | us-west-2 |
|---------|-----------|-----------|
| c7i.8xlarge pricing | ~$1.36/hr | ~$1.44/hr |
| Availability | Excellent | Excellent |
| Best for | East Coast, Global | West Coast |
| Latency (SSH) | Lower from East | Lower from West |

**Recommendation**:
- If you're in North America → **us-east-1** (save ~$0.08/hr)
- If you're on West Coast and want lower SSH latency → **us-west-2**
- Either is fine for benchmarking (latency doesn't matter for batch workloads)

## Step-by-Step Deployment

### Step 1: AWS CLI Login

```bash
# Configure AWS CLI (one-time setup)
aws configure

# You'll be prompted for:
# - AWS Access Key ID: [Your access key]
# - AWS Secret Access Key: [Your secret key]
# - Default region name: us-east-1  (or us-west-2)
# - Default output format: json
```

**Verify authentication:**
```bash
# Check if you're logged in
aws sts get-caller-identity

# Expected output:
# {
#     "UserId": "...",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/yourname"
# }
```

### Step 2: Customize Configuration

Edit `terraform/variables.tf` to personalize your deployment:

```bash
cd terraform
nano variables.tf  # or use your preferred editor
```

**Required changes:**
```hcl
variable "project_prefix" {
  default     = "yourname-emb-bench"  # CHANGE THIS - make it unique!
}

variable "owner" {
  default     = "yourname"  # CHANGE THIS
}

variable "aws_region" {
  default     = "us-east-1"  # Or us-west-2
}

variable "allowed_ssh_cidr" {
  # IMPORTANT: Restrict to your IP for security!
  default     = ["YOUR.IP.ADDRESS.HERE/32"]
  # To get your IP: curl ifconfig.me
}
```

**Get your IP address:**
```bash
curl ifconfig.me
# Use this in allowed_ssh_cidr: ["YOUR.IP/32"]
```

### Step 3: Update AMI for Your Region

If using **us-east-1**: No change needed (default AMI is set)

If using **us-west-2** or another region:
```bash
# Find latest Ubuntu 22.04 AMI for your region
aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name,CreationDate]' \
  --output table

# Update variables.tf with the AMI ID
```

### Step 4: Initialize Terraform

```bash
cd terraform

# Initialize Terraform
terraform init

# Expected output:
# Terraform has been successfully initialized!
```

### Step 5: Plan Deployment

```bash
# Preview what will be created
terraform plan

# Review the output - you should see:
# - aws_security_group.benchmark_sg
# - aws_key_pair.benchmark_key
# - aws_instance.benchmark_instance
# - aws_eip.benchmark_eip
```

### Step 6: Deploy Infrastructure

```bash
# Deploy!
terraform apply

# You'll be prompted: "Do you want to perform these actions?"
# Type: yes
```

**Expected output:**
```
Apply complete! Resources: 4 added, 0 changed, 0 destroyed.

Outputs:

instance_details = {
  "availability_zone" = "us-east-1a"
  "instance_id" = "i-0123456789abcdef0"
  "instance_type" = "c7i.8xlarge"
  "private_ip" = "172.31.x.x"
  "public_ip" = "54.x.x.x"
}
instance_id = "i-0123456789abcdef0"
instance_public_ip = "54.x.x.x"
instance_type = "c7i.8xlarge"
ssh_command = "ssh -i ~/.ssh/id_rsa ubuntu@54.x.x.x"
```

### Step 7: Save Connection Info

```bash
# Save the instance IP
export INSTANCE_IP=$(terraform output -raw instance_public_ip)

# Display connection info
echo "Instance IP: $INSTANCE_IP"
echo "SSH Command: ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP"

# Save to file for later
echo "export INSTANCE_IP=$INSTANCE_IP" > ../instance_info.sh
```

### Step 8: Wait for Instance Setup

```bash
# The instance needs 3-5 minutes to complete initial setup
# Check if instance is ready:
aws ec2 describe-instance-status \
  --instance-ids $(terraform output -raw instance_id) \
  --query 'InstanceStatuses[0].InstanceStatus.Status' \
  --output text

# Wait until output shows: ok
```

### Step 9: Connect to Instance

```bash
# SSH into the instance
ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP

# If you get "Connection refused", wait another minute and retry
```

### Step 10: Setup on Instance

Once connected to the instance:

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/embedding_model_benchmarking.git
cd embedding_model_benchmarking

# Run setup script
bash scripts/setup.sh

# This will:
# - Create directories
# - Install Python dependencies
# - Pull Docker base images
# - Verify Docker is working
```

## Troubleshooting

### "Permission denied (publickey)"
```bash
# Ensure your SSH key is loaded
ssh-add ~/.ssh/id_rsa

# Or specify key explicitly
ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP
```

### "Connection timed out"
```bash
# Check security group allows your IP
aws ec2 describe-security-groups \
  --group-ids $(terraform output -json instance_details | jq -r '.security_group_ids[0]')

# Update allowed_ssh_cidr in variables.tf if needed
```

### Instance not starting
```bash
# Check instance status
aws ec2 describe-instances \
  --instance-ids $(terraform output -raw instance_id) \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text

# Should show: running
```

## Cost Management

### Monitor Costs
```bash
# Check current costs (requires Cost Explorer enabled)
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost
```

### Stop Instance (when not in use)
```bash
# Stop instance (keeps data, stops billing for compute)
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)

# Start instance (when needed again)
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)
```

### Destroy Infrastructure (when done)
```bash
cd terraform

# Destroy all resources
terraform destroy

# Type: yes to confirm

# This will:
# - Terminate EC2 instance
# - Release Elastic IP
# - Delete security group
# - Delete SSH key pair
```

## Expected Costs

**c7i.8xlarge pricing:**
- us-east-1: ~$1.36/hour
- us-west-2: ~$1.44/hour

**Storage:**
- 100GB gp3: ~$8/month (~$0.011/hour)

**Total hourly cost:**
- us-east-1: ~$1.37/hour
- us-west-2: ~$1.45/hour

**Typical benchmark run:**
- Duration: 2-4 hours
- Cost: ~$3-6 per complete run (all 3 frameworks)

## Next Steps

After deployment, proceed to model download and benchmarking:

1. Download model: `python3 scripts/download_model.py --model embeddinggemma-300m`
2. Convert to ONNX: `python3 scripts/convert_to_onnx.py --model embeddinggemma-300m`
3. Convert to OpenVINO: `python3 scripts/convert_to_openvino.py --model embeddinggemma-300m`
4. Build containers: `docker-compose build`
5. Run benchmarks: `python3 orchestrator/runner.py --model embeddinggemma-300m`

See `README.md` for detailed usage instructions.
