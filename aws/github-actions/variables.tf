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

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "github_org" {
  description = "GitHub organization or username"
  type        = string
  default     = "KristinaKuzmenko"
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
  default     = "maritime-qa-assistant"
}

variable "ecr_repository_name" {
  description = "ECR repository name"
  type        = string
  default     = "maritime-qa-app"
}
