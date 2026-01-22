# =============================================================================
# Infrastructure Module Outputs
# =============================================================================

# -----------------------------------------------------------------------------
# EKS
# -----------------------------------------------------------------------------

output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "kubeconfig_command" {
  value = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# -----------------------------------------------------------------------------
# ECR
# -----------------------------------------------------------------------------

output "ecr_backend_url" {
  value = module.ecr.backend_repository_url
}

output "ecr_app_url" {
  value = module.ecr.backend_repository_url
}

# -----------------------------------------------------------------------------
# IAM Roles
# -----------------------------------------------------------------------------

output "app_role_arn" {
  value = aws_iam_role.app.arn
}

output "qdrant_role_arn" {
  value = aws_iam_role.qdrant.arn
}

output "external_secrets_role_arn" {
  value = aws_iam_role.external_secrets.arn
}

output "alb_controller_role_arn" {
  value       = aws_iam_role.alb_controller.arn
  description = "IAM role ARN for AWS Load Balancer Controller"
}

# -----------------------------------------------------------------------------
# S3 (from storage module)
# -----------------------------------------------------------------------------

output "s3_app_bucket" {
  value = local.s3_app_bucket_name
}

output "s3_backup_bucket" {
  value = local.s3_backup_bucket_name
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

output "summary" {
  value = <<-EOT

================================================================================
Maritime QA - Infrastructure Deployed!
================================================================================

☸️  EKS Cluster:
   Name: ${module.eks.cluster_name}
   Endpoint: ${module.eks.cluster_endpoint}

🐳 ECR:
  App: ${module.ecr.backend_repository_url}

📦 S3 Buckets (from storage module):
   App Files: ${local.s3_app_bucket_name}
   Backups:   ${local.s3_backup_bucket_name}

🔐 IAM Roles:
   App:    ${aws_iam_role.app.arn}
   Qdrant: ${aws_iam_role.qdrant.arn}
  ExternalSecrets: ${aws_iam_role.external_secrets.arn}

📋 Next Steps:

1. Configure kubectl:
   ${module.eks.kubeconfig_command_with_region}

2. Apply generated manifests:
   kubectl apply -f generated/k8s-config.yaml

3. Create/update AWS Secrets Manager secret (example):
  aws secretsmanager put-secret-value --region ${var.aws_region} --secret-id ${var.project_name}/${var.environment}/app --secret-string '{"OPENAI_API_KEY":"...","NEO4J_URI":"...","NEO4J_PASSWORD":"..."}'

4. Install External Secrets Operator (uses IRSA ServiceAccount created above):
  helm repo add external-secrets https://charts.external-secrets.io
  helm upgrade --install external-secrets external-secrets/external-secrets -n ${var.external_secrets_namespace} --create-namespace --set serviceAccount.create=false --set serviceAccount.name=${var.external_secrets_service_account}

5. Build and push app:
   docker build -t ${module.ecr.backend_repository_url}:latest .
   docker push ${module.ecr.backend_repository_url}:latest

6. Deploy app:
  kubectl apply -k maritime-qa-assistant/k8s/base

================================================================================
EOT
}
