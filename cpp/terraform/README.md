# Terraform Configuration for C++ Embedding Benchmarks

This directory contains Terraform configurations to deploy an AWS EC2 instance optimized for C++ embedding model benchmarking.

## Overview

This Terraform setup creates:
- **EC2 Instance**: c7i.8xlarge (32 vCPU, 64GB RAM, Intel Xeon Sapphire Rapids)
- **Elastic IP**: Static public IP for consistent access
- **Security Group**: SSH (port 22) and HTTP (port 8000) access
- **Optimized Setup**: Performance CPU governor, disabled swap, THP disabled
- **Pre-installed**: ONNX Runtime C++, OpenVINO C++, build tools, Python

## Why c7i.8xlarge?

The **Intel c7i instances** (Sapphire Rapids) are ideal for C++ benchmarking:
- **OpenVINO optimization**: Intel-specific optimizations work best on Intel CPUs
- **AVX-512 support**: Advanced vector extensions for faster inference
- **32 vCPU**: Sufficient parallelism for multi-threaded inference
- **64GB RAM**: Enough for large models + build artifacts
- **DDR5 memory**: Lower latency, higher bandwidth

**Cost**: ~$1.36/hour (~$32/day if running 24/7)

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured with credentials:
   ```bash
   aws configure
   # Or use AWS SSO:
   aws sso login --profile your-profile
   export AWS_PROFILE=your-profile
   ```
3. **Terraform** installed (>= 1.0):
   ```bash
   # macOS
   brew install terraform

   # Linux
   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
   unzip terraform_1.6.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   ```
4. **SSH key pair** at `~/.ssh/id_rsa.pub` (or customize path in `variables.tf`)

## Quick Start

### 1. Configure Variables

Edit `variables.tf` and customize:

```hcl
variable "project_prefix" {
  default     = "your-name-cpp-bench"  # CHANGE THIS
}

variable "owner" {
  default     = "your-name"  # CHANGE THIS
}

variable "allowed_ssh_cidr" {
  default     = ["YOUR.IP.ADDRESS/32"]  # CHANGE THIS for security
}
```

### 2. Initialize Terraform

```bash
cd cpp/terraform
terraform init
```

### 3. Preview Changes

```bash
terraform plan
```

Review the resources that will be created.

### 4. Deploy

```bash
terraform apply
```

Type `yes` when prompted. Deployment takes ~3-5 minutes.

### 5. Get Connection Info

```bash
terraform output
```

You'll see:
- Public IP address
- SSH command
- Server URLs
- Quick start guide

### 6. SSH into Instance

```bash
# Wait ~2-3 minutes for user_data setup to complete
ssh -i ~/.ssh/id_rsa ubuntu@$(terraform output -raw public_ip)

# Check setup progress
tail -f /var/log/user-data.log

# Wait for "Setup complete!" message
```

### 7. Build and Run

Once logged in:

```bash
# Clone your repository
git clone <your-repo-url> ~/benchmark
cd ~/benchmark/cpp

# Build servers (dependencies already installed!)
cd onnx_runtime && ./build.sh && cd ..
cd openvino && ./build.sh && cd ..

# Download and convert models
cd ..
python3 scripts/download_model.py --model embeddinggemma-300m
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
python3 scripts/convert_to_openvino.py --model embeddinggemma-300m

# Run benchmarks
cd cpp
./benchmark_comparison.py --model embeddinggemma-300m --iterations 1000
```

## Terraform Commands

### View Outputs

```bash
terraform output
terraform output public_ip
terraform output ssh_command
```

### Update Infrastructure

After modifying `.tf` files:

```bash
terraform plan
terraform apply
```

### Destroy Infrastructure

**WARNING: This will delete the instance and all data!**

```bash
terraform destroy
```

Type `yes` to confirm.

### SSH into Instance

```bash
terraform output -raw ssh_command | bash
```

Or:

```bash
ssh -i ~/.ssh/id_rsa ubuntu@$(terraform output -raw public_ip)
```

## What Gets Installed

The `user_data.sh` script automatically installs:

### System Packages
- Build tools: gcc, g++, make, cmake
- Development libraries: libssl-dev, yaml-cpp
- Utilities: git, curl, wget, vim, htop, jq

### C++ Inference Frameworks
- **ONNX Runtime 1.17.1** (system-wide in `/usr/local/onnxruntime/`)
- **OpenVINO 2024.0.0** (in `/opt/intel/openvino_2024/`)

### Third-party C++ Libraries
- cpp-httplib (HTTP server, header-only)
- nlohmann/json (JSON parser, header-only)
- tokenizers-cpp (fast tokenizer with bindings)

### Python Environment
- Python 3.11
- PyTorch 2.2.0 (CPU)
- Transformers 4.38.0
- ONNX Runtime 1.17.1
- OpenVINO tools 2024.0.0

### System Optimizations
- CPU governor set to `performance`
- Swap disabled
- Transparent Huge Pages (THP) disabled
- Increased file descriptor limits
- Kernel network tuning

## Systemd Services

Two systemd services are pre-configured (disabled by default):

### ONNX Runtime C++ Service

```bash
# After building, enable and start:
sudo systemctl enable onnx-cpp-server
sudo systemctl start onnx-cpp-server

# Check status:
sudo systemctl status onnx-cpp-server

# View logs:
sudo journalctl -u onnx-cpp-server -f
```

### OpenVINO C++ Service

```bash
# After building, enable and start:
sudo systemctl enable openvino-cpp-server
sudo systemctl start openvino-cpp-server

# Check status:
sudo systemctl status openvino-cpp-server

# View logs:
sudo journalctl -u openvino-cpp-server -f
```

## Cost Optimization

### Instance Options

```hcl
# variables.tf

# Production benchmarking (default)
variable "instance_type" {
  default = "c7i.8xlarge"  # $1.36/hour, 32 vCPU, 64GB RAM
}

# Development/testing (cheaper)
variable "instance_type" {
  default = "c7i.4xlarge"  # $0.68/hour, 16 vCPU, 32GB RAM
}

# Small workloads
variable "instance_type" {
  default = "c7i.2xlarge"  # $0.34/hour, 8 vCPU, 16GB RAM
}
```

### Stop Instance When Not in Use

```bash
# Stop (preserves data, stops billing for compute)
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)

# Start again
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)

# Note: Elastic IP continues to incur charges (~$0.005/hour when not attached)
```

### Delete When Done

```bash
terraform destroy  # Deletes everything, stops all charges
```

## Security Best Practices

### 1. Restrict SSH Access

Edit `variables.tf`:

```hcl
variable "allowed_ssh_cidr" {
  # Find your IP: curl ifconfig.me
  default = ["YOUR.IP.ADDRESS/32"]  # Only your IP
}
```

### 2. Use SSH Keys

The default uses `~/.ssh/id_rsa.pub`. To use a different key:

```hcl
variable "ssh_public_key_path" {
  default = "~/.ssh/your-custom-key.pub"
}
```

### 3. Use AWS SSO/Profiles

```bash
export AWS_PROFILE=your-profile-name
terraform apply
```

### 4. Tag Resources

All resources are tagged with:
- `Owner`: Your name
- `Project`: embedding-benchmarking-cpp
- `ManagedBy`: terraform
- `Environment`: benchmark-cpp

## Troubleshooting

### Issue: `terraform apply` fails with permission errors

**Solution**: Check AWS credentials:
```bash
aws sts get-caller-identity
```

### Issue: SSH connection refused

**Solution**: Wait for user_data setup to complete (~3 minutes):
```bash
aws ec2 get-console-output --instance-id $(terraform output -raw instance_id)
```

### Issue: Build fails with missing libraries

**Solution**: Check user_data setup logs:
```bash
ssh ubuntu@$(terraform output -raw public_ip)
tail -100 /var/log/user-data.log
```

### Issue: Instance type not available

**Solution**: Choose different region or instance type:
```hcl
variable "aws_region" {
  default = "us-west-2"  # Try different region
}

variable "instance_type" {
  default = "c7i.4xlarge"  # Try smaller instance
}
```

### Issue: Can't destroy - resources in use

**Solution**: Force destroy:
```bash
# Stop instance first
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)

# Then destroy
terraform destroy -auto-approve
```

## Advanced Configuration

### Custom AMI

To use a different Ubuntu version:

```bash
# Find latest Ubuntu 22.04 AMI in your region
aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text

# Update variables.tf
variable "ami_id" {
  default = "ami-xxxxxxxxx"  # Your AMI
}
```

### Larger Storage

```hcl
variable "root_volume_size" {
  default = 200  # 200 GB (default is 100 GB)
}
```

### Multiple Instances

To run multiple benchmarks in parallel:

```bash
# Create workspace per instance
terraform workspace new benchmark-1
terraform apply

terraform workspace new benchmark-2
terraform apply

# List workspaces
terraform workspace list

# Switch between them
terraform workspace select benchmark-1
```

## Files in This Directory

- `main.tf` - Main Terraform configuration (EC2, security, networking)
- `variables.tf` - Configurable variables (instance type, region, etc.)
- `outputs.tf` - Output values (IP, SSH command, URLs)
- `user_data.sh` - Initialization script (runs on first boot)
- `README.md` - This file

## Next Steps

After deployment:

1. **Build servers** - See `../README.md`
2. **Download models** - Use scripts in `../../scripts/`
3. **Run benchmarks** - Use `../benchmark_comparison.py`
4. **Monitor performance** - Use `htop`, `top`, systemd logs
5. **Destroy when done** - `terraform destroy` to avoid charges

## Support

For issues:
- Check `/var/log/user-data.log` on the instance
- Review Terraform logs
- Consult AWS EC2 documentation
- File an issue in the repository

---

**Cost Reminder**: c7i.8xlarge costs ~$1.36/hour. Remember to destroy resources when done!
