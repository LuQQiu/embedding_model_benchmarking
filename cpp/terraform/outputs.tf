output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.cpp_benchmark_instance.id
}

output "public_ip" {
  description = "Public IP address (Elastic IP)"
  value       = aws_eip.cpp_benchmark_eip.public_ip
}

output "public_dns" {
  description = "Public DNS name"
  value       = aws_instance.cpp_benchmark_instance.public_dns
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_eip.cpp_benchmark_eip.public_ip}"
}

output "instance_type" {
  description = "Instance type"
  value       = aws_instance.cpp_benchmark_instance.instance_type
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "onnx_server_url" {
  description = "ONNX Runtime C++ server URL (once started)"
  value       = "http://${aws_eip.cpp_benchmark_eip.public_ip}:8000"
}

output "openvino_server_url" {
  description = "OpenVINO C++ server URL (once started)"
  value       = "http://${aws_eip.cpp_benchmark_eip.public_ip}:8000"
}

output "setup_status" {
  description = "Command to check setup status"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_eip.cpp_benchmark_eip.public_ip} 'tail -f /var/log/user-data.log'"
}

output "quick_start" {
  description = "Quick start commands"
  value = <<-EOT

  ========================================
  C++ Benchmark Instance Created!
  ========================================

  1. SSH into instance:
     ${format("ssh -i ~/.ssh/id_rsa ubuntu@%s", aws_eip.cpp_benchmark_eip.public_ip)}

  2. Check setup progress:
     tail -f /var/log/user-data.log

  3. Wait for setup to complete (look for "Setup complete!" message)

  4. Clone repository and prepare:
     git clone <your-repo-url> ~/benchmark
     cd ~/benchmark

  5. Setup C++ dependencies:
     cd cpp
     ./setup.sh
     ./install_onnxruntime.sh
     ./install_openvino.sh
     source /opt/intel/openvino_2024/setupvars.sh

  6. Build servers:
     cd onnx_runtime && ./build.sh && cd ..
     cd openvino && ./build.sh && cd ..

  7. Download and convert models:
     cd ..
     python3 scripts/download_model.py --model embeddinggemma-300m
     python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
     python3 scripts/convert_to_openvino.py --model embeddinggemma-300m

  8. Run benchmark:
     cd cpp
     ./benchmark_comparison.py --model embeddinggemma-300m --iterations 1000

  Instance IP: ${aws_eip.cpp_benchmark_eip.public_ip}
  Instance Type: ${aws_instance.cpp_benchmark_instance.instance_type}
  CPU: 32 vCPU Intel Xeon 8488C (Sapphire Rapids)
  RAM: 64 GB DDR5

  ========================================
  EOT
}
