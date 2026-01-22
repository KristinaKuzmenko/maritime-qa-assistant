# =============================================================================
# EKS Module Outputs
# =============================================================================

# -----------------------------------------------------------------------------
# Cluster Outputs
# -----------------------------------------------------------------------------

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.this.name
}

output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.this.id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.this.arn
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = aws_eks_cluster.this.endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded CA certificate for cluster"
  value       = aws_eks_cluster.this.certificate_authority[0].data
}

output "cluster_version" {
  description = "Kubernetes version"
  value       = aws_eks_cluster.this.version
}

# -----------------------------------------------------------------------------
# OIDC Outputs (CRITICAL for IRSA)
# -----------------------------------------------------------------------------

output "oidc_provider_arn" {
  description = "ARN of the OIDC Provider for IRSA"
  value       = aws_iam_openid_connect_provider.eks.arn
}

output "oidc_provider_url" {
  description = "URL of the OIDC Provider (without https://)"
  value       = replace(aws_eks_cluster.this.identity[0].oidc[0].issuer, "https://", "")
}

output "oidc_issuer" {
  description = "OIDC issuer URL (with https://)"
  value       = aws_eks_cluster.this.identity[0].oidc[0].issuer
}

# -----------------------------------------------------------------------------
# Security Group Outputs
# -----------------------------------------------------------------------------

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
}

output "cluster_primary_security_group_id" {
  description = "Primary security group ID created by EKS"
  value       = aws_security_group.cluster.id
}

# -----------------------------------------------------------------------------
# Node Group Outputs
# -----------------------------------------------------------------------------

output "node_group_name" {
  description = "Name of the node group"
  value       = aws_eks_node_group.this.node_group_name
}

output "node_role_arn" {
  description = "ARN of the IAM role for nodes"
  value       = aws_iam_role.node.arn
}

output "node_role_name" {
  description = "Name of the IAM role for nodes"
  value       = aws_iam_role.node.name
}

# -----------------------------------------------------------------------------
# kubeconfig Helper
# -----------------------------------------------------------------------------

output "kubeconfig_command" {
  description = "Command to update kubeconfig"
  value       = "aws eks update-kubeconfig --region ${data.aws_caller_identity.current.account_id} --name ${aws_eks_cluster.this.name}"
}

# Get region from ARN
data "aws_region" "current" {}

output "kubeconfig_command_with_region" {
  description = "Command to update kubeconfig with region"
  value       = "aws eks update-kubeconfig --region ${data.aws_region.current.name} --name ${aws_eks_cluster.this.name}"
}
