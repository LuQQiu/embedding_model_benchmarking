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
}

# Security group for SSH access and HTTP (for testing)
resource "aws_security_group" "cpp_benchmark_sg" {
  name        = "${var.project_prefix}-cpp-sg"
  description = "Security group for ${var.project_prefix} C++ embedding model benchmarking instance"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidr
  }

  ingress {
    description = "HTTP for testing"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidr
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_prefix}-cpp-sg"
    Project     = "embedding-benchmarking-cpp"
    Owner       = var.owner
    Environment = var.environment
    ManagedBy   = "terraform"
    Purpose     = "Security group for C++ benchmark instance"
  }
}

# SSH key pair
resource "aws_key_pair" "cpp_benchmark_key" {
  key_name   = "${var.project_prefix}-cpp-key"
  public_key = file(var.ssh_public_key_path)

  tags = {
    Name        = "${var.project_prefix}-cpp-key"
    Project     = "embedding-benchmarking-cpp"
    Owner       = var.owner
    Environment = var.environment
    ManagedBy   = "terraform"
    Purpose     = "SSH key for C++ benchmark instance"
  }
}

# EC2 instance - Intel c7i.8xlarge (optimized for C++ and OpenVINO)
resource "aws_instance" "cpp_benchmark_instance" {
  ami           = var.ami_id
  instance_type = var.instance_type

  key_name               = aws_key_pair.cpp_benchmark_key.key_name
  vpc_security_group_ids = [aws_security_group.cpp_benchmark_sg.id]

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 125
    delete_on_termination = true

    tags = {
      Name        = "${var.project_prefix}-cpp-root-volume"
      Project     = "embedding-benchmarking-cpp"
      Owner       = var.owner
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }

  user_data = file("${path.module}/user_data.sh")

  tags = {
    Name        = "${var.project_prefix}-cpp-instance"
    Project     = "embedding-benchmarking-cpp"
    Owner       = var.owner
    Environment = var.environment
    ManagedBy   = "terraform"
    Purpose     = "C++ embedding model benchmarking (ONNX Runtime vs OpenVINO)"
    InstanceType = var.instance_type
    CPU         = "32 vCPU (Intel Xeon 8488C Sapphire Rapids)"
    Memory      = "64 GB DDR5"
    Description = "Native C++ benchmarking: ONNX Runtime C++ vs OpenVINO C++"
    Frameworks  = "onnx-cpp, openvino-cpp"
  }

  # Don't recreate if user_data changes (for manual updates)
  lifecycle {
    ignore_changes = [user_data]
  }
}

# Elastic IP for consistent access
resource "aws_eip" "cpp_benchmark_eip" {
  instance = aws_instance.cpp_benchmark_instance.id
  domain   = "vpc"

  tags = {
    Name        = "${var.project_prefix}-cpp-eip"
    Project     = "embedding-benchmarking-cpp"
    Owner       = var.owner
    Environment = var.environment
    ManagedBy   = "terraform"
    Purpose     = "Static IP for C++ benchmark instance"
  }
}
