variable "project_prefix" {
  description = "Prefix for all resource names to distinguish from other resources"
  type        = string
  default     = "lu-emb-bench" # CHANGE THIS to your unique identifier
}

variable "owner" {
  description = "Owner tag for all resources"
  type        = string
  default     = "lu" # CHANGE THIS to your name
}

variable "aws_region" {
  description = "AWS region to deploy the benchmark instance"
  type        = string
  default     = "us-east-1" # Recommended: us-east-1 (best availability) or us-west-2
}

variable "ami_id" {
  description = "AMI ID for Ubuntu 22.04 LTS"
  type        = string
  # Ubuntu 22.04 LTS in us-east-1 (updated 2024)
  # Find latest: aws ec2 describe-images --owners 099720109477 --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text
  default = "ami-0e86e20dae9224db8"

  # Other regions (uncomment the one you need):
  # us-west-2: ami-0075013580f6322a1
  # eu-west-1: ami-0694d931cee176e7d
}

variable "ssh_public_key_path" {
  description = "Path to your SSH public key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "allowed_ssh_cidr" {
  description = "CIDR blocks allowed to SSH into the instance"
  type        = list(string)
  default     = ["0.0.0.0/0"] # WARNING: Change to your IP for security! e.g., ["1.2.3.4/32"]
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c7i.8xlarge" # 32 vCPU, 64GB RAM, Intel Xeon 8488C
}

variable "root_volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 100
}

variable "environment" {
  description = "Environment tag"
  type        = string
  default     = "benchmark"
}
