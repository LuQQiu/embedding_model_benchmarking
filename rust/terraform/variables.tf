variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "ami_id" {
  description = "AMI ID for Ubuntu 22.04 LTS (update for your region)"
  type        = string
  # Ubuntu 22.04 LTS (Jammy) for us-west-2
  # Find latest: aws ec2 describe-images --owners 099720109477 --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" --query 'sort_by(Images, &CreationDate)[-1].ImageId'
  default     = "ami-0cf2b4e024cdb6960"  # Update for your region
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c7i.16xlarge"  # 64 vCPUs, 128 GB RAM, Intel Sapphire Rapids
}

variable "key_name" {
  description = "SSH key pair name"
  type        = string
  default     = "id_rsa"  # Update with your key name
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH to the instance"
  type        = string
  default     = "0.0.0.0/0"  # WARNING: Restrict this in production!
}

variable "root_volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 200
}
