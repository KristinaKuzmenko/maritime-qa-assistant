# =============================================================================
# VPC Module Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.this.id
}

output "vpc_cidr" {
  description = "VPC CIDR block"
  value       = aws_vpc.this.cidr_block
}

output "public_subnets" {
  description = "List of public subnet IDs"
  value       = [aws_subnet.public_1.id, aws_subnet.public_2.id]
}

output "private_subnets" {
  description = "List of private subnet IDs"
  value       = [aws_subnet.private_1.id, aws_subnet.private_2.id]
}

output "public_subnet_cidrs" {
  description = "List of public subnet CIDRs"
  value       = [aws_subnet.public_1.cidr_block, aws_subnet.public_2.cidr_block]
}

output "private_subnet_cidrs" {
  description = "List of private subnet CIDRs"
  value       = [aws_subnet.private_1.cidr_block, aws_subnet.private_2.cidr_block]
}

output "internet_gateway_id" {
  description = "Internet Gateway ID"
  value       = aws_internet_gateway.this.id
}

output "nat_gateway_id" {
  description = "NAT Gateway ID"
  value       = var.enable_nat_gateway ? aws_nat_gateway.this[0].id : null
}

output "nat_gateway_public_ip" {
  description = "NAT Gateway public IP"
  value       = var.enable_nat_gateway ? aws_eip.nat[0].public_ip : null
}
