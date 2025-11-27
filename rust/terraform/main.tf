terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "embedding-benchmarks-rust"
      Environment = "development"
      ManagedBy   = "terraform"
    }
  }
}

# SSH key pair
resource "aws_key_pair" "rust_benchmark_key" {
  key_name   = "rust-benchmark-key"
  public_key = file("~/.ssh/id_rsa.pub")

  tags = {
    Name = "rust-benchmark-key"
  }
}

# Security Group
resource "aws_security_group" "rust_benchmark_sg" {
  name        = "rust-benchmark-sg"
  description = "Allow SSH and HTTP inbound traffic"

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH access"
  }

  # HTTP (for debugging)
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "HTTP API access for debugging"
  }

  # All outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "rust-benchmark-sg"
  }
}

# EC2 Instance
resource "aws_instance" "rust_benchmark" {
  ami           = var.ami_id
  instance_type = var.instance_type

  key_name               = aws_key_pair.rust_benchmark_key.key_name
  vpc_security_group_ids = [aws_security_group.rust_benchmark_sg.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    delete_on_termination = true

    tags = {
      Name = "rust-benchmark-root"
    }
  }

  user_data = file("${path.module}/user_data.sh")

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  tags = {
    Name = "rust-embedding-benchmark"
  }
}

# Elastic IP
resource "aws_eip" "rust_benchmark_eip" {
  instance = aws_instance.rust_benchmark.id
  domain   = "vpc"

  tags = {
    Name = "rust-benchmark-eip"
  }
}
