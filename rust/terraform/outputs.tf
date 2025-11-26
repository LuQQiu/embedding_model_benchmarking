output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.rust_benchmark.id
}

output "instance_public_ip" {
  description = "Public IP address of the instance"
  value       = aws_eip.rust_benchmark_eip.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the instance"
  value       = aws_eip.rust_benchmark_eip.public_dns
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_name} ubuntu@${aws_eip.rust_benchmark_eip.public_ip}"
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.rust_benchmark_sg.id
}
