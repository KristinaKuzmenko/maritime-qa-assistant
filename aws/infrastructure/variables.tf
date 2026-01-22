# =============================================================================
# Infrastructure Module Variables
# =============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "maritime-qa"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# -----------------------------------------------------------------------------
# VPC
# -----------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zone" {
  description = "First AZ"
  type        = string
  default     = "us-east-1a"
}

variable "availability_zone_2" {
  description = "Second AZ"
  type        = string
  default     = "us-east-1b"
}

variable "public_subnet_cidr" {
  default = "10.0.1.0/24"
}

variable "public_subnet_cidr_2" {
  default = "10.0.3.0/24"
}

variable "private_subnet_cidr" {
  default = "10.0.2.0/24"
}

variable "private_subnet_cidr_2" {
  default = "10.0.4.0/24"
}

# -----------------------------------------------------------------------------
# EKS
# -----------------------------------------------------------------------------

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "maritime-qa-dev"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.31"
}

variable "node_instance_types" {
  description = "EC2 instance types for nodes"
  type        = list(string)
  default     = ["m7i-flex.large"]
}

variable "node_desired_size" {
  type    = number
  default = 2
}

variable "node_min_size" {
  type    = number
  default = 1
}

variable "node_max_size" {
  type    = number
  default = 4
}

variable "node_disk_size" {
  type    = number
  default = 50
}

variable "use_spot_instances" {
  type    = bool
  default = false
}

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------

variable "app_namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "maritime-qa"
}

variable "app_service_account" {
  description = "Kubernetes ServiceAccount name"
  type        = string
  default     = "maritime-qa-app"
}

variable "qdrant_host" {
  description = "Qdrant hostname. Leave empty to use in-cluster service name. For Qdrant Cloud set to the cloud hostname."
  type        = string
  default     = ""
}

variable "qdrant_port" {
  description = "Qdrant port (usually 6333)."
  type        = number
  default     = 6333
}

variable "external_secrets_namespace" {
  description = "Namespace where External Secrets Operator runs"
  type        = string
  default     = "external-secrets"
}

variable "external_secrets_service_account" {
  description = "ServiceAccount name used by External Secrets Operator"
  type        = string
  default     = "external-secrets"
}
