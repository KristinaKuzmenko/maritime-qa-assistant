# =============================================================================
# IAM Module Variables
# =============================================================================

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "oidc_provider_arn" {
  description = "ARN of the EKS OIDC provider"
  type        = string
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

# -----------------------------------------------------------------------------
# S3 Access
# -----------------------------------------------------------------------------

variable "s3_bucket_arn" {
  description = "ARN of S3 bucket for app storage"
  type        = string
}

# -----------------------------------------------------------------------------
# Service Account Configuration
# -----------------------------------------------------------------------------

variable "app_namespace" {
  description = "Kubernetes namespace for the application"
  type        = string
  default     = "default"
}

variable "app_service_account" {
  description = "Kubernetes service account name for the app"
  type        = string
  default     = "maritime-qa-app"
}

variable "qdrant_service_account" {
  description = "Kubernetes service account name for Qdrant"
  type        = string
  default     = "qdrant"
}

# -----------------------------------------------------------------------------
# Optional Features
# -----------------------------------------------------------------------------

variable "create_qdrant_role" {
  description = "Create IRSA role for Qdrant (for EBS snapshots)"
  type        = bool
  default     = false
}

variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs access for app"
  type        = bool
  default     = true
}

variable "node_role_name" {
  description = "Name of EKS node IAM role (for additional policies)"
  type        = string
  default     = ""
}
