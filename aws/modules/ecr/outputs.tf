# =============================================================================
# ECR Module Outputs
# =============================================================================

output "backend_repository_url" {
  description = "App ECR repository URL (legacy name: backend_repository_url)"
  value       = aws_ecr_repository.backend.repository_url
}

output "backend_repository_arn" {
  description = "App ECR repository ARN (legacy name: backend_repository_arn)"
  value       = aws_ecr_repository.backend.arn
}

output "backend_repository_name" {
  description = "App ECR repository name (legacy name: backend_repository_name)"
  value       = aws_ecr_repository.backend.name
}

output "qdrant_repository_url" {
  description = "Qdrant ECR repository URL"
  value       = var.create_qdrant_repo ? aws_ecr_repository.qdrant[0].repository_url : null
}

output "registry_id" {
  description = "ECR registry ID (AWS account ID)"
  value       = aws_ecr_repository.backend.registry_id
}

# Docker login command helper
output "docker_login_command" {
  description = "AWS CLI command to login to ECR"
  value       = "aws ecr get-login-password --region ${data.aws_region.current.name} | docker login --username AWS --password-stdin ${aws_ecr_repository.backend.registry_id}.dkr.ecr.${data.aws_region.current.name}.amazonaws.com"
}

data "aws_region" "current" {}
