output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.benchmark_instance.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_eip.benchmark_eip.public_ip
}

output "instance_type" {
  description = "Instance type"
  value       = aws_instance.benchmark_instance.instance_type
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_eip.benchmark_eip.public_ip}"
}

output "instance_details" {
  description = "Full instance details"
  value = {
    instance_id   = aws_instance.benchmark_instance.id
    instance_type = aws_instance.benchmark_instance.instance_type
    public_ip     = aws_eip.benchmark_eip.public_ip
    private_ip    = aws_instance.benchmark_instance.private_ip
    availability_zone = aws_instance.benchmark_instance.availability_zone
  }
}
