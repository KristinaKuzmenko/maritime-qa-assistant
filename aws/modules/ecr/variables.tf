# =============================================================================
# ECR Module Variables
# =============================================================================

variable "project_name" {
  description = "Project name for repository naming"
  type        = string
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

# -----------------------------------------------------------------------------
# Repository Options
# -----------------------------------------------------------------------------

variable "create_qdrant_repo" {
  description = "Create ECR repository for custom Qdrant image"
  type        = bool
  default     = false
}

variable "image_tag_mutability" {
  description = "Image tag mutability (MUTABLE or IMMUTABLE)"
  type        = string
  default     = "MUTABLE"
}

variable "scan_on_push" {
  description = "Enable image scanning on push"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# Lifecycle Policy
# -----------------------------------------------------------------------------

variable "keep_image_count" {
  description = "Number of images to keep"
  type        = number
  default     = 10
}

variable "untagged_image_days" {
  description = "Days to keep untagged images"
  type        = number
  default     = 7
}

# -----------------------------------------------------------------------------
# Access Policy
# -----------------------------------------------------------------------------

variable "eks_node_role_arn" {
  description = "ARN of EKS node IAM role (for pull access)"
  type        = string
  default     = ""
}
