# Quick Start - AWS Deployment

## TL;DR - Fastest Path to Deployment

```bash
# 1. Login to AWS (one-time setup)
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Format (json)

# 2. Run automated deployment
./deploy.sh

# 3. Follow the prompts - it will:
#    ✓ Check prerequisites
#    ✓ Get your IP for security
#    ✓ Deploy infrastructure
#    ✓ Wait for instance ready
#    ✓ Show SSH command

# 4. SSH into instance (shown in output)
ssh -i ~/.ssh/id_rsa ubuntu@<INSTANCE_IP>

# 5. Setup on instance
git clone <your-repo>
cd embedding_model_benchmarking
bash scripts/setup.sh

# 6. Run benchmarks
python3 scripts/download_model.py --model embeddinggemma-300m
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
python3 scripts/convert_to_openvino.py --model embeddinggemma-300m
docker-compose build
python3 orchestrator/runner.py --model embeddinggemma-300m

# 7. When done - DESTROY to stop billing!
cd terraform && terraform destroy
```

## Region Comparison

| Region | Pros | Cons | Hourly Cost |
|--------|------|------|-------------|
| **us-east-1** ✅ | Lowest cost, best availability | Higher latency from West Coast | ~$1.36/hr |
| **us-west-2** | Lower latency (West Coast) | ~6% higher cost | ~$1.44/hr |

**Recommendation**: Use **us-east-1** unless you need lower SSH latency from West Coast.

## Resource Naming

All AWS resources will be tagged with:
- **Name**: `{your-prefix}-{resource-type}` (e.g., `lu-emb-bench-instance`)
- **Owner**: Your name
- **Project**: `embedding-benchmarking`
- **ManagedBy**: `terraform`

This makes it easy to:
- Find your resources in AWS console
- Distinguish from other projects
- Track costs by owner/project
- Clean up everything with `terraform destroy`

## Cost Warning

The c7i.8xlarge instance costs **~$1.36-1.44/hour**.

**Always destroy when not in use:**
```bash
cd terraform
terraform destroy
```

Or stop the instance (keeps data but stops compute billing):
```bash
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)
```

## Files Created

After running `./deploy.sh`, you'll have:
- `terraform/terraform.tfvars` - Your configuration
- `instance_info.sh` - Connection details (source with `. instance_info.sh`)

## Troubleshooting

**"AWS credentials not configured"**
```bash
aws configure
```

**"Permission denied (publickey)"**
```bash
ssh-add ~/.ssh/id_rsa
# Or specify key: ssh -i ~/.ssh/id_rsa ubuntu@IP
```

**"Connection timeout"**
- Wait 2-3 minutes for instance to boot
- Check security group allows your IP
- Verify instance is running: `aws ec2 describe-instances`

## Manual Deployment (if script fails)

```bash
cd terraform

# Edit variables.tf with your details
nano variables.tf

# Deploy
terraform init
terraform plan
terraform apply

# Get connection info
terraform output ssh_command
```

See `DEPLOYMENT.md` for detailed step-by-step instructions.
