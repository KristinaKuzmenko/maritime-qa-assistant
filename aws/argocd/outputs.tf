output "argocd_namespace" {
  value       = kubernetes_namespace.argo.metadata[0].name
  description = "Namespace Argo CD"
}

output "argocd_release_name" {
  value       = helm_release.argo.name
  description = "Helm release name"
}

output "argocd_release_status" {
  value       = helm_release.argo.status
  description = "Helm release status"
}

# output "argocd_application_name" {
#   value       = kubernetes_manifest.maritime_qa_app.manifest["metadata"]["name"]
#   description = "Argo CD Application created by Terraform"
# }
