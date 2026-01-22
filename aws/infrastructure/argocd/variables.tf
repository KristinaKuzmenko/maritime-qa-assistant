variable "aws_region" {
  default     = "us-east-1"
  description = "Default Region"
}

variable "eks_state_bucket" {
 description = "S3 bucket with remote state EKS"
 type    = string
 default   = "mlops-tfstate-kristinakuzmenko"
}

variable "eks_state_key" {
 description = "S3 key for remote state EKS"
 type    = string
 # Must match aws/infrastructure backend key
 default   = "maritime-qa/infrastructure/terraform.tfstate"
}

variable "eks_state_region" {
 description = "Region for bucket with remote state EKS"
 type    = string
 default   = "eu-central-1"
}

variable "argocd_namespace" {
 description = "Namespace for Argo CD"
 type    = string
 default   = "infra-tools"
}

variable "argocd_chart_version" {
 description = "Helm-chart version for Argo CD"
 type    = string
 default   = "7.7.5"
}

variable "app_repo_url" {
 description = "Public Git-repo with manifests"
 type    = string
 default   = "https://github.com/KristinaKuzmenko/maritime-qa-assistant.git"
}

variable "app_repo_branch" {
 description = "Branch"
 type    = string
 default   = "main"
}

variable "argocd_app_name" {
 description = "Argo CD Application name for this project"
 type        = string
 default     = "maritime-qa"
}

variable "app_kustomize_path" {
 description = "Path in the repo for the Argo CD Application (Kustomize entrypoint)"
 type        = string
 default     = "maritime-qa-assistant/k8s/base"
}

variable "app_destination_namespace" {
 description = "Kubernetes namespace where the app will be deployed"
 type        = string
 default     = "maritime-qa"
}