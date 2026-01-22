# =============================================================================
# EKS Module Variables
# =============================================================================

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.32"
}

variable "vpc_id" {
  description = "VPC ID where cluster will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for EKS cluster and nodes"
  type        = list(string)
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# -----------------------------------------------------------------------------
# Node Group Configuration
# -----------------------------------------------------------------------------

variable "node_instance_types" {
  description = "Instance types for node group"
  type        = list(string)
  default     = ["m7i-flex.large"]
}

variable "node_desired_size" {
  description = "Desired number of nodes"
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum number of nodes"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes"
  type        = number
  default     = 4
}

variable "node_disk_size" {
  description = "Disk size for nodes (GB)"
  type        = number
  default     = 50
}

variable "use_spot_instances" {
  description = "Use spot instances for cost savings"
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# Cluster Configuration
# -----------------------------------------------------------------------------

variable "enabled_log_types" {
  description = "List of EKS control plane log types to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator"]
}

variable "ebs_csi_version" {
  description = "Version of EBS CSI driver addon"
  type        = string
  default     = null  # Uses latest compatible version
}
