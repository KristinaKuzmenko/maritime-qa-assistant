# =============================================================================
# IAM Module Outputs
# =============================================================================

output "app_role_arn" {
  description = "ARN of the app IRSA role"
  value       = aws_iam_role.app.arn
}

output "app_role_name" {
  description = "Name of the app IRSA role"
  value       = aws_iam_role.app.name
}

output "qdrant_role_arn" {
  description = "ARN of the Qdrant IRSA role"
  value       = var.create_qdrant_role ? aws_iam_role.qdrant[0].arn : null
}

output "qdrant_role_name" {
  description = "Name of the Qdrant IRSA role"
  value       = var.create_qdrant_role ? aws_iam_role.qdrant[0].name : null
}

# Kubernetes manifest helpers
output "app_service_account_yaml" {
  description = "YAML for Kubernetes ServiceAccount with IRSA annotation"
  value       = <<-EOT
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${var.app_service_account}
  namespace: ${var.app_namespace}
  annotations:
    eks.amazonaws.com/role-arn: ${aws_iam_role.app.arn}
EOT
}

output "qdrant_service_account_yaml" {
  description = "YAML for Qdrant ServiceAccount with IRSA annotation"
  value = var.create_qdrant_role ? <<-EOT
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${var.qdrant_service_account}
  namespace: ${var.app_namespace}
  annotations:
    eks.amazonaws.com/role-arn: ${aws_iam_role.qdrant[0].arn}
EOT
  : null
}
